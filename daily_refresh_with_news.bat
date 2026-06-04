@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run run_app.bat once first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python scanner.py --days 120 --mode top500 --with-news --news-limit 100
pause
