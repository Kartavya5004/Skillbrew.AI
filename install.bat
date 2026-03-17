@echo off
SETLOCAL
echo.
echo =====================================================
echo   Skillbrew.AI v2 -- Windows Setup
echo =====================================================
echo.

python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Python not found. & pause & exit /b 1)

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [1/6] Creating virtual environment...
    python -m venv .venv
) ELSE ( echo [1/6] venv already exists. )

call .venv\Scripts\activate.bat

echo [2/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [3/6] Installing packages (one by one)...
pip install python-dotenv>=1.0.0  --quiet
pip install numpy>=1.24.0          --quiet
pip install opencv-python>=4.8.0   --quiet
pip install mediapipe>=0.10.30     --quiet
pip install scikit-learn>=1.3.0    --quiet
pip install joblib>=1.3.0          --quiet
pip install pandas>=2.0.0          --quiet
pip install matplotlib>=3.7.0      --quiet
pip install seaborn>=0.12.0        --quiet
pip install tqdm>=4.65.0           --quiet
pip install flask>=3.0.0           --quiet
pip install flask-cors>=4.0.0      --quiet
pip install flask-socketio>=5.3.0  --quiet
pip install eventlet>=0.35.0       --quiet
pip install librosa>=0.10.0        --quiet
pip install sounddevice>=0.4.6     --quiet
pip install soundfile>=0.12.1      --quiet
pip install kaggle>=1.6.0          --quiet

echo [4/6] Copying .env template...
IF NOT EXIST ".env" ( copy .env.example .env >nul )

echo [5/6] Downloading MediaPipe model...
python scripts/download_model.py

echo [6/6] Done!
echo.
echo =====================================================
echo   NEXT STEPS:
echo   1. Edit .env  ^(add KAGGLE_USERNAME + KAGGLE_KEY^)
echo   2. python scripts/setup_kaggle.py
echo   3. python train_model.py       ^(~5 min^)
echo   4. python app.py               ^(starts server^)
echo   5. Open http://127.0.0.1:5000
echo =====================================================
echo.
pause
