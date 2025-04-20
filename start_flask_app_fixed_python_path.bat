@echo off
cd /d %~dp0
echo Starting Flask server...
start "" /b cmd /c "C:/Users/Aswin/AppData/Local/Programs/Python/Python313/python.exe inference.py"
timeout /t 3 >nul
start http://127.0.0.1:5000
pause