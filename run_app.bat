@echo off
setlocal
cd /d %~dp0

where py >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Python launcher py was not found. Install Python 3.10/3.11 and tick Add to PATH.
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment with py -3...
  py -3 -m venv .venv
)

call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo Installing requirements...
python -m pip install -r requirements.txt

echo Starting SwingNSE Desktop...
python -m streamlit run app.py
pause
