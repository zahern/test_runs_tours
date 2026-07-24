@echo off
REM One-click launcher for the Model Constraint Editor.
REM Double-click this file. It finds Python, installs streamlit if needed,
REM and opens the editor in your browser. No build step required.
setlocal
cd /d "%~dp0"

REM Prefer an explicit interpreter if the repo env is active, else fall back.
where python >nul 2>&1 && (set "PY=python") || (
  where py >nul 2>&1 && (set "PY=py") || (set "PY=python")
)

echo Launching the Model Constraint Editor with %PY% ...
"%PY%" "%~dp0launch_constraint_gui.py"
if errorlevel 1 (
  echo.
  echo The launcher exited with an error. See the messages above.
  pause
)
endlocal
