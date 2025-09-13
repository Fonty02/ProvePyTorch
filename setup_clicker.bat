@echo off
echo ====================================
echo   Auto Clicker Setup per Windows
echo ====================================
echo.

echo Controllo se Python e installato...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRORE: Python non e installato o non e nel PATH.
    echo Scarica Python da: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Python trovato!
echo.

echo Installazione delle dipendenze...
echo Installazione di pyautogui...
pip install pyautogui

if %errorlevel% neq 0 (
    echo.
    echo ERRORE: Installazione fallita.
    echo Prova a eseguire questo script come amministratore.
    echo.
    pause
    exit /b 1
)

echo.
echo ====================================
echo   Installazione completata!
echo ====================================
echo.
echo Per usare l'auto clicker, esegui:
echo   python mouse_clicker.py
echo.
echo IMPORTANTE:
echo - Posiziona il mouse dove vuoi che avvengano i clic
echo - Premi CTRL+C per fermare lo script
echo - Lo script iniziera dopo un countdown di 3 secondi
echo.
pause
