@echo off
cd /d "%~dp0"
echo Starting Stellar Logic AI Dashboard...
echo Executive Dashboard: http://localhost:5000/dashboard.html
echo Assistant Dashboard: http://localhost:5000/assistant.html
echo.
python dashboard_server.py
pause
