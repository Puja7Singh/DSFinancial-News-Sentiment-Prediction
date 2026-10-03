@echo off
REM Double-click (Windows) to install dependencies and launch the dashboard.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
