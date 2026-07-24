@echo off
REM build_beamer.bat — Compile the Beamer presentation
set SRC=Z:\test_runs_tours\abm_safety\beamer
set TMP=C:\Users\ahernz\AppData\Local\Temp\opencode\beamer_build

echo Cleaning temp...
rmdir /s /q "%TMP%" 2>nul
mkdir "%TMP%"
mkdir "%TMP%\figures"
mkdir "%TMP%\interactive"

echo Copying files...
copy "%SRC%\model_review_beamer.tex" "%TMP%\" >nul
xcopy "%SRC%\figures\*" "%TMP%\figures\" /Y /Q >nul
copy "%SRC%\interactive\map.html" "%TMP%\interactive\" >nul 2>nul

echo Compiling (pass 1)...
cd /d "%TMP%"
pdflatex -interaction=nonstopmode model_review_beamer.tex >nul
echo Compiling (pass 2)...
pdflatex -interaction=nonstopmode model_review_beamer.tex >nul

echo Copying back...
copy "%TMP%\model_review_beamer.pdf" "%SRC%\model_review_beamer.pdf" /Y

echo.
echo ============================================
echo   Done! PDF: model_review_beamer.pdf
echo ============================================
pause
