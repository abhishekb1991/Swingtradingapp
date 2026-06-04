@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run run_app.bat once first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python scanner.py --news-only --news-limit 100
pause
