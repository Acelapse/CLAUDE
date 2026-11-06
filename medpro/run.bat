@echo off
cd /d "%~dp0src"
call ..\venv\Scripts\activate.bat
python manage.py runserver 8000
