@echo off
cd /d %~dp0
py -3 -c "from swingnse.learning_engine import update_outcomes, train_learning_model; print('Updated outcomes:', update_outcomes()); print(train_learning_model(min_rows=80))"
pause
