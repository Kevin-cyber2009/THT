@echo off
REM build.bat - Build client to standalone .exe

echo ============================================================
echo   BUILD DEEPFAKE DETECTOR CLIENT
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11+
    pause
    exit /b 1
)

echo [INFO] Python found!
echo.

REM Check PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Check if client.py exists
if not exist "client.py" (
    echo [ERROR] client.py not found!
    echo Please make sure client.py is in the same directory
    pause
    exit /b 1
)

echo [INFO] Building executable...
echo [INFO] This may take 2-5 minutes...
echo.

REM Build with PyInstaller
pyinstaller --onefile ^
    --windowed ^
    --name="DeepfakeDetectorClient" ^
    --icon=icon.ico ^
    --add-data="icon.ico;." ^
    client.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Check errors above
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo.
echo Output: dist\DeepfakeDetectorClient.exe
echo.

REM Show file info
if exist "dist\DeepfakeDetectorClient.exe" (
    echo File info:
    dir dist\DeepfakeDetectorClient.exe | find "DeepfakeDetectorClient.exe"
    echo.
    echo Size: ~5-10MB (lightweight!)
    echo.
)

echo ============================================================
echo   NEXT STEPS
echo ============================================================
echo.
echo 1. Test:
echo    dist\DeepfakeDetectorClient.exe
echo.
echo 2. Configure API URL:
echo    Settings tab ^> Enter your server URL
echo    Example: https://your-app.onrender.com
echo.
echo 3. Distribute:
echo    Share dist\DeepfakeDetectorClient.exe to users
echo.
echo NOTE: Users need internet to connect to your API server
echo ============================================================
echo.

pause
