#!/usr/bin/env python3
"""
Auto Clicker Semplice per Windows
Script minimale per clic ripetuti del mouse

Uso: python simple_clicker.py
"""

try:
    import pyautogui
    import time
    print("Auto Clicker - Posiziona il mouse e premi Invio...")
    input()
    
    print("Inizio tra 3 secondi...")
    time.sleep(3)
    
    print("Clic in corso! Premi CTRL+C per fermare.")
    
    try:
        while True:
            pyautogui.click()
            time.sleep(0.0000000001)  # 10 clic al secondo
    except KeyboardInterrupt:
        print("\nScript fermato.")
        
except ImportError:
    print("Installa pyautogui con: pip install pyautogui")
except Exception as e:
    print(f"Errore: {e}")

input("Premi Invio per uscire...")
