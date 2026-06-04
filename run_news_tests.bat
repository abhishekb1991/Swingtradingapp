@echo off
cd /d %~dp0
py -3 tests\test_news_engine.py
pause
