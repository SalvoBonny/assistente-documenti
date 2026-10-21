@echo off
rem File per avviare l'app con un doppio clic.

rem Si sposta nella cartella in cui si trova questo file
cd /d "%~dp0"

rem Avvia Streamlit usando Python dell'ambiente virtuale .venv
.\.venv\Scripts\python.exe -m streamlit run app.py

rem Se qualcosa va storto, la finestra resta aperta cosi' puoi leggere l'errore
pause
