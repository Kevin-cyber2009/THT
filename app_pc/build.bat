@echo off
REM build_modern.bat - Build modern client

echo ============================================================
echo   BUILD MODERN DEEPFAKE DETECTOR CLIENT
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

REM Check PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
)

REM Check client file
if not exist "client.py" (
    echo [ERROR] client.py not found!
    pause
    exit /b 1
)

echo [INFO] Building modern client...
echo.

REM Build
pyinstaller --onefile ^
    --windowed ^
    --name="DeepfakeDetector" ^
    --icon=icon.ico ^
    client.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo.
echo Output: dist\DeepfakeDetector.exe
echo Size: ~5-10MB
echo.
echo IMPORTANT: Before distributing, update RENDER_API_URL in client.py
echo            Line 24: RENDER_API_URL = "https://your-app.onrender.com"
echo.
echo Then rebuild: build_modern.bat
echo ============================================================
echo.

pause