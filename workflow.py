import torch
from torch import nn #modulo che contiene le funzioni per la costruzione dei modelli di rete neurale
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42) #per rendere i risultati riproducibili

#1 -> Predisporre i dati

#2 -> Costruire il modello

#3 -> Addestrare il modello

#4 -> Valutare il modello e fare predizioni

#5 -> Salvare il modello

#6 -> Mettere tutto insieme


#ESEMPIO MOLTO SEMPLICE -> Predirre il valore di una funzione lineare (retta)


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10,6))
    plt.scatter(train_data.cpu(), train_labels.cpu(), c="b", label="Training data")
    plt.scatter(test_data.cpu(), test_labels.cpu(), c="g", label="Testing data")
    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c="r", label="Predictions")
    plt.legend()
    plt.show()


# 1) Predisporre i dati

weight=0.7
bias=0.3

start=0
end=1
step=0.02
X=torch.arange(start,end,step,device=device).unsqueeze(1) #il vettore sarebbee [0,0.002,0.004,0.006,...,0.998]. Con unsqueeze(0) diventa [[0,0.002,0.004,0.006,...,0.998]]  mentre con unsqueeze(1) diventa [[0],[0.002],[0.004],[0.006],...,[0.998]]
#le due feature da apprendere sono bias e weight, quindi le X rappresentano il "termine noto" della feature 1 (seconda) di ogni esperimento (le feature 0 è come se fosse sempre = 1)
y=weight*X+bias
train_split=int(0.8*len(X))
X_train,y_train, X_test, y_test = X[:train_split], y[:train_split], X[train_split:], y[train_split:]


# 2) Costruire il modello

class LinearRegression(nn.Module): #OGNI MODELLO DEVE ESSERE UNA SOTTOCLASSE DI nn.Module
    def __init__(self): #costruttore
        super().__init__() #richiama il costruttore della superclasse
        self.weights=nn.Parameter(torch.randn(1,
                                              requires_grad=True,
                                              dtype=torch.float)) #per creare i parametri random
        self.bias=nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor: #metodo che definisce come i dati vengono passati attraverso il modello
        return self.weights*x+self.bias
    


model=LinearRegression().to(device) #istanzio il modello e lo sposto sulla GPU se disponibile
print(model.weights, model.bias)
loss_fn=nn.L1Loss() #funzione di loss (errore) -> mean(|y_true-y_pred|)
optimizer=torch.optim.SGD(model.parameters(), lr=0.0001) #ottimizzatore -> Stochastic Gradient Descent con learning rate=0.1

# 3) Addestrare il modello
epochs=18001

#tracking dei valori
epoch_count=[]
loss_values=[]
test_loss_values=[]


for epoch in range(epochs):
    model.train() #mette il modello in modalità di addestramento
    predictions=model(X_train) #predizione
    loss=loss_fn(predictions, y_train) #calcolo dell'errore -> predizioni, valori reali
    optimizer.zero_grad() #azzera i gradienti
    loss.backward() #calcolo del gradiente
    optimizer.step() #aggiornamento dei pesi
    model.eval() #mette il modello in modalità di valutazione
    with torch.inference_mode():
        test_predictions=model(X_test) #predizione
        test_loss=loss_fn(test_predictions, y_test) #calcolo dell'errore
    if epoch%100==0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"Epoch {epoch}: Loss {loss.item()}, Test Loss {test_loss.item()}")

plt.plot(epoch_count, loss_values, label="Training Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.legend()
plt.title("Training and test loss")
plt.show()

#4) FARE PREDIZIONI
model.eval() #mette il modello in modalità di valutazione
with torch.inference_mode(): #disabilita il calcolo del gradiente. Esiste anche torch.no_grad() ma torch.inference_mode() è più efficiente perchè disabilita il tracciamento di autograd (cioè riabilitare il calcolo in un momento successivo)
    predictions=model(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=predictions)


#5) SALVARE IL MODELLO
torch.save(model.state_dict(), "model.pth") #salva i pesi

#6) RICALCARLO in model_1
model_1=LinearRegression().to(device)
model_1.load_state_dict(torch.load("model.pth"))
model_1.eval()
with torch.inference_mode():
    predictions=model_1(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=predictions)

torch.manual_seed(42)
#versione migliore
class  LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model_v2=LinearRegressionV2().to(device)
loss_fn=nn.L1Loss()
optimizer=torch.optim.SGD(model_v2.parameters(), lr=0.0001)

# 3) Addestrare il modello
epochs=18001

#tracking dei valori
epoch_count=[]
loss_values=[]
test_loss_values=[]

for epoch in range(epochs):
    model_v2.train()
    predictions=model_v2(X_train)
    loss=loss_fn(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_v2.eval()
    with torch.inference_mode():
        test_predictions=model_v2(X_test)
        test_loss=loss_fn(test_predictions, y_test)
    if epoch%100==0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"Epoch {epoch}: Loss {loss.item()}, Test Loss {test_loss.item()}")
plt.plot(epoch_count, loss_values, label="Training Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.legend()
plt.title("Training and test loss")
plt.show()
model_v2.eval()
with torch.inference_mode():
    predictions=model_v2(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=predictions)
