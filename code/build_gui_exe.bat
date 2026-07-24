@echo off
REM Build a standalone Windows .exe for the Model Constraint Editor launcher.
REM
REM The .exe is a THIN launcher: it locates your Python, ensures streamlit is
REM present, and runs `streamlit run constraint_gui.py`. Streamlit itself is NOT
REM frozen (that path is fragile), so the .exe must sit next to constraint_gui.py,
REM constraint_schema.json and build_constraint_schema.py — i.e. keep it in this
REM code folder. Distribute the folder, not the .exe alone.
REM
REM Usage:  double-click, or run  build_gui_exe.bat
setlocal
cd /d "%~dp0"

where python >nul 2>&1 && (set "PY=python") || (set "PY=py")

echo Installing PyInstaller if needed...
"%PY%" -m pip install --quiet pyinstaller || goto :err

echo Building ConstraintGUI.exe ...
"%PY%" -m PyInstaller --onefile --console --name ConstraintGUI ^
    --distpath "%~dp0" --workpath "%~dp0build_exe" --specpath "%~dp0build_exe" ^
    "%~dp0launch_constraint_gui.py" || goto :err

echo.
echo Done. Created ConstraintGUI.exe in this folder.
echo Double-click ConstraintGUI.exe (keep it beside constraint_gui.py).
goto :eof

:err
echo.
echo Build failed. You can still use "Launch Constraint GUI.bat" (no build needed).
pause
