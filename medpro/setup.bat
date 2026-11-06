@echo off
REM Windows: downloads Med-Pro, applies the fixes in patch\, installs everything.
cd /d "%~dp0"
if not exist src git clone --depth 1 https://github.com/kumar11jr/Med_pro_webApp.git src || goto :err
copy /Y patch\views.py src\medpro\views.py >nul
copy /Y patch\templates\disout.html src\templates\disout.html >nul
if not exist venv py -3 -m venv venv || goto :err
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
cd src && python manage.py migrate -v0 && cd ..
echo.
echo Done. Start it with run.bat, then open http://localhost:8000
pause
exit /b 0
:err
echo Setup failed. See the error above.
pause
exit /b 1
