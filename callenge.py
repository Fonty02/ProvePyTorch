import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
from anomaly_detection.config import DATA_PROCESSED_DIR, MODELS_DIR


# ----------------- Preprocessing -----------------
def load_and_preprocess(train_file, test_file):
    """
    Carica i dataset train/test, separa features e target, e codifica le etichette.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    feature_cols = [c for c in train_df.columns if c not in ["TRACE_ID", "ANOMALY_CATEGORY"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["ANOMALY_CATEGORY"].values

    le = LabelEncoder()
    y_int = le.fit_transform(y_train)

    X_test = test_df[feature_cols].values

    return X_train, y_int, X_test, le, test_df


# ----------------- Custom weighted geometric mean scorer -----------------
def weighted_geometric_mean(y_true, y_pred):
    """
    Calcola il geometric mean pesato dei recall per ogni classe.
    Protegge da recall=0 usando epsilon.
    """
    recalls = recall_score(y_true, y_pred, average=None)
    recalls = np.clip(recalls, 1e-6, 1)  # evita che 0 annulli tutto
    weights = np.bincount(y_true)
    weights = weights / weights.sum()
    return np.prod(np.power(recalls, weights))


wgm_scorer = make_scorer(weighted_geometric_mean, greater_is_better=True)


# ----------------- First Grid Search -----------------
def first_grid_search(X, y, random_seed=42):
    """
    Grid search sui parametri principali di CatBoost.
    """
    param_grid = {
        'iterations': [1000],
        'learning_rate': [0.1],
        'depth': [8]
    }

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=random_seed,
        thread_count=-1,
        verbose=100
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

    print(f"\n🔎 Prima grid search: {len(param_grid['iterations']) * len(param_grid['learning_rate']) * len(param_grid['depth'])} combinazioni totali")

    grid_search = GridSearchCV(model, param_grid, scoring=wgm_scorer, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    print("\n📌 Best parameters from first grid search:", grid_search.best_params_)
    print("📌 Best weighted geometric mean:", grid_search.best_score_)

    return grid_search.best_estimator_


# ----------------- Second Grid Search on Class Weights -----------------
def second_grid_search_class_weights(X, y, base_model, class_weights_grid, random_seed=42, target_score=None):
    """
    Grid search sui pesi delle classi con verbosity avanzata e possibilità di specificare
    i valori da provare per ogni classe.

    Parameters
    ----------
    X, y : ndarray
        Dati di training.
    base_model : CatBoostClassifier
        Modello base già ottimizzato dai parametri principali.
    class_weights_grid : dict
        Dizionario {classe_id: [lista di pesi possibili]}.
        Esempio: {0: [1, 1.2], 1: [0.8, 1], 2: [1, 1.5]}
    random_seed : int
        Seed di riproducibilità.
    target_score : float, opzionale
        Se non None, interrompe la grid search quando viene superata questa soglia.
    """
    n_classes = len(np.unique(y))

    # Costruisci tutte le combinazioni dai pesi specificati per ogni classe
    keys = sorted(class_weights_grid.keys())
    values = [class_weights_grid[k] for k in keys]
    class_weight_combinations = [list(w) for w in product(*values)]

    print(f"\n🔎 Second grid search: {len(class_weight_combinations)} combinazioni di pesi da testare "
          f"per {n_classes} classi.")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

    best_score = -np.inf
    best_weights = None
    best_model = None

    for i, weights in enumerate(class_weight_combinations, 1):
        print(f"\n➡️  Fit {i}/{len(class_weight_combinations)} con pesi: {weights}")

        model = CatBoostClassifier(
            **base_model.get_params(),
            class_weights=weights,
        )

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            score = weighted_geometric_mean(y[val_idx], preds)
            fold_scores.append(score)
            print(f"   Fold {fold}: WGM={score:.4f}")

        mean_score = np.mean(fold_scores)
        print(f"   ✅ Mean WGM = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_weights = weights
            best_model = model

        # Early stopping manuale
        if target_score is not None and mean_score >= target_score:
            print(f"\n⏹ Early stopping: superata la soglia {target_score} con WGM={mean_score:.4f}")
            break

    print("\n📌 Best class weights:", best_weights)
    print("📌 Best weighted geometric mean with class weights:", best_score)

    return best_model


# ----------------- Main -----------------
if __name__ == "__main__":
    # Percorsi dataset
    train_path = os.path.join(DATA_PROCESSED_DIR, "train", "train_standard.csv")
    test_path = os.path.join(DATA_PROCESSED_DIR, "test", "test_standard.csv")

    # Caricamento dati
    X_train, y_train, X_test, le, test_df = load_and_preprocess(train_path, test_path)

    # --- Prima Grid Search ---
    print("\n🔹 Starting first grid search on main parameters...")
    best_model = first_grid_search(X_train, y_train)

    # --- Seconda Grid Search sui pesi delle classi ---
    print("\n🔹 Starting second grid search on class weights...")

    # 👇 qui definisci tu i pesi da provare per ciascuna classe
    class_weights_grid = {
        0: [0.7, 0.8],   # Classe 0
        1: [0.7, 0.8],   # Classe 1
        2: [0.7, 0.8, 1],    # Classe 2
        3: [1],    # Classe 3
        4: [1]    # Classe 4
    }

    best_model_with_weights = second_grid_search_class_weights(
        X_train, y_train,
        base_model=best_model,
        class_weights_grid=class_weights_grid,
        target_score=0.90  # fermati se superi 0.90 di WGM
    )

    # --- Training finale sul dataset completo ---
    print("\n🔹 Training final model on full data...")
    best_model_with_weights.fit(X_train, y_train)

    # --- Predizioni test ---
    y_test_pred_int = best_model_with_weights.predict(X_test).ravel()
    y_test_pred_labels = le.inverse_transform(y_test_pred_int.astype(int))

    # --- Salvataggio predizioni ---
    pred_data = pd.DataFrame({
        "TRACE_ID": test_df["TRACE_ID"],
        "ANOMALY_CATEGORY": y_test_pred_labels
    })
    pred_file = os.path.join(MODELS_DIR, "porcelli", "standard", "catboost_gridsearch_predictions.dsv")
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    pred_data.to_csv(pred_file, index=False, sep=";")
    print(f"\n📂 Predictions saved to {pred_file}")

    # --- Salvataggio modello finale ---
    model_file = os.path.join(MODELS_DIR, "porcelli", "standard", "catboost_gridsearch_model.cbm")
    best_model_with_weights.save_model(model_file)
    print(f"📂 Final model saved to {model_file}")
