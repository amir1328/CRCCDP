echo Installing requirements (This may take a while)...
pip install streamlit pandas pillow tensorflow numpy scipy
echo.

if not exist model.h5 (
    echo Model not found! Starting training process...
    echo It will take some time to train the model.
    python train.py
) else (
    echo Model found. Skipping training.
)

echo Starting the web application...
streamlit run app.py
pause
