echo Cleaning up conflicting PyTorch installations...
pip uninstall -y torch torchvision torchaudio ultralytics

echo Installing PyTorch (CUDA 11.8) with --no-cache-dir to ensure clean files...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --force-reinstall

echo Installing other dependencies...
pip install streamlit pandas pillow tensorflow numpy scipy ultralytics
echo.

if not exist yolo_results\weights\best.pt (
    echo YOLO model not found! Starting YOLOv8 training...
    echo This might take a while...
    python train_yolo.py
) else (
    echo YOLO model found.
)

if not exist model.h5 (
    echo MobileNet model not found! Starting Legacy training...
    python train.py
) else (
    echo MobileNet model found.
)

echo Starting the web application...
streamlit run app.py
pause
