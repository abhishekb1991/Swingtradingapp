@echo off
cd /d %~dp0
py -3 -c "from swingnse.learning_engine import train_learning_model; print(train_learning_model(min_rows=80))"
pause
