@echo off
REM Local MTC benchmark launcher — uses the venv in this folder.
cd /d "%~dp0"
".venv\Scripts\python.exe" -u run_mtc_benchmark_local.py %*
echo exit_code=%ERRORLEVEL%
pause
