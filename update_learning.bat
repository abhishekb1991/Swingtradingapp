@echo off
cd /d %~dp0
py -3 -c "from swingnse.learning_engine import update_outcomes; print('Updated outcomes:', update_outcomes())"
pause
