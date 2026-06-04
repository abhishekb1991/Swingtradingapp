@echo off
cd /d %~dp0
py -3 scanner.py --macro-only
pause
