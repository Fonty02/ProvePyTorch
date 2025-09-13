#!/usr/bin/env python3
"""
Script per Windows - Auto Clicker del Mouse
Questo script simula clic ripetuti del tasto sinistro del mouse.

Controlli:
- Premi CTRL+C per fermare lo script
- Modifica le variabili per personalizzare il comportamento

Requisiti: pip install pyautogui
"""

import pyautogui
import time
import sys
import signal

def signal_handler(sig, frame):
    """Gestisce l'interruzione dello script con CTRL+C"""
    print("\n\nScript interrotto dall'utente.")
    sys.exit(0)

def auto_clicker():
    """Funzione principale per l'auto-clicker"""
    
    # Configurazioni personalizzabili
    DELAY_BETWEEN_CLICKS = 0.0000000001  # Secondi tra ogni clic (0.1 = 10 clic al secondo)
    CLICK_COUNT = 10000000000  # Numero totale di clic (0 = infinito)
    COUNTDOWN_SECONDS = 3  # Secondi di countdown prima di iniziare
    
    # Registra il gestore per CTRL+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== AUTO CLICKER DEL MOUSE ===")
    print(f"Configurazione:")
    print(f"- Ritardo tra clic: {DELAY_BETWEEN_CLICKS} secondi")
    print(f"- Numero di clic: {'Infinito' if CLICK_COUNT == 0 else CLICK_COUNT}")
    print(f"- Premi CTRL+C per fermare lo script")
    print()
    
    # Countdown prima di iniziare
    print("Posiziona il mouse dove vuoi che avvengano i clic...")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Inizio tra {i} secondi...", end='\r')
        time.sleep(1)
    print("INIZIO!")
    print()
    
    # Ottieni la posizione iniziale del mouse
    start_x, start_y = pyautogui.position()
    print(f"Posizione di clic: ({start_x}, {start_y})")
    print("Clic in corso... (CTRL+C per fermare)")
    print()
    
    try:
        click_counter = 0
        while True:
            # Esegui il clic
            pyautogui.click(start_x, start_y)
            click_counter += 1
            
            # Mostra il progresso
            print(f"Clic eseguiti: {click_counter}", end='\r')
            
            # Controlla se abbiamo raggiunto il limite di clic
            if CLICK_COUNT > 0 and click_counter >= CLICK_COUNT:
                print(f"\nCompletati {CLICK_COUNT} clic!")
                break
            
            # Attendi prima del prossimo clic
            time.sleep(DELAY_BETWEEN_CLICKS)
            
    except KeyboardInterrupt:
        print(f"\n\nScript interrotto dopo {click_counter} clic.")
    except Exception as e:
        print(f"\nErrore: {e}")
    
    print("Script terminato.")

def interactive_mode():
    """Modalità interattiva per configurare i parametri"""
    print("=== CONFIGURAZIONE INTERATTIVA ===")
    
    try:
        # Chiedi i parametri all'utente
        delay = input("Ritardo tra clic in secondi (default 0.1): ").strip()
        if not delay:
            delay = 0.1
        else:
            delay = float(delay)
        
        count = input("Numero di clic (0 per infinito, default 100): ").strip()
        if not count:
            count = 100
        else:
            count = int(count)
        
        countdown = input("Secondi di countdown (default 3): ").strip()
        if not countdown:
            countdown = 3
        else:
            countdown = int(countdown)
        
        return delay, count, countdown
        
    except ValueError:
        print("Valore non valido inserito. Uso valori di default.")
        return 0.1, 100, 3

if __name__ == "__main__":
    try:
        # Controlla se pyautogui è installato
        import pyautogui
        
        # Disabilita il fail-safe di pyautogui (movimento del mouse agli angoli)
        pyautogui.FAILSAFE = True
        
        print("Auto Clicker del Mouse per Windows")
        print("=" * 35)
        
        # Chiedi se usare modalità interattiva
        mode = input("Vuoi configurare i parametri? (s/n, default n): ").strip().lower()
        
        if mode == 's' or mode == 'si':
            delay, count, countdown = interactive_mode()
            
            # Modifica le variabili globalmente
            import mouse_clicker
            mouse_clicker.DELAY_BETWEEN_CLICKS = delay
            mouse_clicker.CLICK_COUNT = count
            mouse_clicker.COUNTDOWN_SECONDS = countdown
        
        auto_clicker()
        
    except ImportError:
        print("ERRORE: La libreria 'pyautogui' non è installata.")
        print("Installa con: pip install pyautogui")
        print()
        input("Premi Invio per uscire...")
    except KeyboardInterrupt:
        print("\nScript interrotto.")
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        input("Premi Invio per uscire...")
