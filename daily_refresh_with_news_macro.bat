@echo off
cd /d %~dp0
py -3 scanner.py --days 120 --mode top500 --with-news --with-macro --news-limit 100
pause
