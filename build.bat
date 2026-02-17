@echo off
REM fix_and_build.bat
REM Uninstall PyQt5, then build final.py to exe

echo ================================================================================
echo FIX PyQt5 CONFLICT + BUILD EXE
echo ================================================================================
echo.

echo [1/4] Uninstalling PyQt5 (conflicts with PySide6)...
pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip PyQt5-plugins PyQt5-tools -y
echo ✓ PyQt5 removed

echo.
echo [2/4] Cleaning old build files...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist *.spec del /q *.spec
echo ✓ Cleaned

echo.
echo [3/4] Building exe from final.py...
echo Please wait 5-10 minutes...
echo.

pyinstaller ^
  --onefile ^
  --windowed ^
  --name="DeepfakeDetector" ^
  --add-data="config.yaml;." ^
  --add-data="models;models" ^
  --add-data="src;src" ^
  --collect-all="PySide6" ^
  final.py

echo.
echo [4/4] Checking result...

if exist "dist\DeepfakeDetector.exe" (
    echo.
    echo ================================================================================
    echo ✓ BUILD SUCCESSFUL!
    echo ================================================================================
    for %%I in ("dist\DeepfakeDetector.exe") do set SIZE=%%~zI
    set /a SIZE_MB=%SIZE%/1048576
    echo File: dist\DeepfakeDetector.exe
    echo Size: %SIZE_MB% MB
    echo.
    set /p RUN="Launch now? (y/n): "
    if /i "%RUN%"=="y" start "" "dist\DeepfakeDetector.exe"
) else (
    echo.
    echo ✗ Still failed. Trying console build to see errors...
    pyinstaller --onefile --console --name="DeepfakeDetector-debug" --collect-all="PySide6" final.py
    echo.
    echo Run this to see the error:
    echo   dist\DeepfakeDetector-debug.exe
)

echo.
pause