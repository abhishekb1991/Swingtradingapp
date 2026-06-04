@echo off
cd /d %~dp0
py -3 scanner.py --days 120 --mode top500 --with-macro
pause
