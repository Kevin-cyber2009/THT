@echo off
echo ============================================================
echo   BUILD OFFLINE PC APP
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
)

if not exist "final.py" (
    echo [ERROR] final.py not found
    pause
    exit /b 1
)

echo [INFO] Building offline desktop app from final.py ...
pyinstaller AIChecker.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL
echo ============================================================
echo Output: dist\AIChecker.exe
echo Mode  : full offline
echo.
pause
