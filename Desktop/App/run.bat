@echo off
call venv\Scripts\activate  REM Activate virtual environment
python app.py  REM Run Flask app
pause  REM Keeps terminal open to check for errors
