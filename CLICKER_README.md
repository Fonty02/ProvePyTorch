# Auto Clicker per Windows

Questo progetto contiene script Python per automatizzare i clic del mouse su Windows.

## File Inclusi

1. **`mouse_clicker.py`** - Script completo con opzioni avanzate
2. **`simple_clicker.py`** - Versione semplificata per uso immediato
3. **`setup_clicker.bat`** - Script di installazione automatica

## Installazione

### Metodo 1: Script Automatico
1. Esegui `setup_clicker.bat` come amministratore
2. Lo script installerà automaticamente le dipendenze

### Metodo 2: Manuale
```bash
pip install pyautogui
```

## Uso

### Script Completo (mouse_clicker.py)
```bash
python mouse_clicker.py
```

Caratteristiche:
- Modalità interattiva per configurare i parametri
- Controllo del numero di clic
- Regolazione della velocità
- Countdown prima dell'inizio
- Interruzione sicura con CTRL+C

### Script Semplice (simple_clicker.py)
```bash
python simple_clicker.py
```

Caratteristiche:
- Uso immediato
- 10 clic al secondo
- Posiziona il mouse e premi Invio per iniziare
- CTRL+C per fermare

## Controlli

- **CTRL+C**: Ferma lo script
- **Fail-safe**: Muovi il mouse nell'angolo in alto a sinistra per fermare pyautogui

## Configurazioni (mouse_clicker.py)

Puoi modificare queste variabili nello script:

```python
DELAY_BETWEEN_CLICKS = 0.1  # Secondi tra clic (0.1 = 10 clic/sec)
CLICK_COUNT = 100          # Numero di clic (0 = infinito)
COUNTDOWN_SECONDS = 3      # Countdown prima dell'inizio
```

## Avvertenze

⚠️ **IMPORTANTE**: 
- Usa responsabilmente questo software
- Non usare per scopi illegali o per aggirare sistemi di sicurezza
- Alcuni giochi e applicazioni possono rilevare l'automazione
- Assicurati di avere il permesso di usare automazione

## Risoluzione Problemi

### "pyautogui non trovato"
```bash
pip install pyautogui
```

### "Permission denied"
- Esegui il prompt dei comandi come amministratore
- O usa: `pip install --user pyautogui`

### Script non si ferma
- Premi CTRL+C
- Oppure muovi il mouse nell'angolo in alto a sinistra dello schermo

## Esempi di Uso

1. **Gaming**: Farming automatico (se permesso dal gioco)
2. **Test**: Stress test di interfacce utente
3. **Automazione**: Clic ripetitivi in applicazioni

## Licenza

Uso libero per scopi educativi e legittimi.
