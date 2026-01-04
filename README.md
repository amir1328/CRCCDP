# CRCCD Explorer & AI Diagnostics

This project contains a **Streamlit web application** and an **AI model training script** for the Colorectal Cancer Classification and Detection (CRCCD) dataset.

## 🚀 Quick Start

Simply double-click the **`run_app.bat`** file in this folder.

This script will automatically:
1.  Install necessary Python dependencies (`tensorflow`, `streamlit`, `pandas`, `pillow`, `scipy`).
2.  **Train the AI Model** (if `model.h5` is not found).
3.  Launch the **Web Application**.

## 📋 Features

### 1. Dataset Overview
- Visualize the distribution of images across all **14 classes** (e.g., Polyps, Ulcerative Colitis, Adenocarcinoma).
- View statistics for both Training and Testing sets.

### 2. Image Gallery
- Browse through the dataset structure interactively.
- Filter images by class type.

### 3. AI Prediction (Diagnostics)
- Uses a **MobileNetV2** Convolutional Neural Network (CNN).
- **Upload Functionality**: Analyze your own medical images.
- **Test Sample**: Pick a random image from the test set to verify the model's performance.
- Displays the **Confidence Score** and probability distribution for the prediction.

## 🛠️ Technical Details

- **`train.py`**: The training pipeline.
    - Uses Transfer Learning with MobileNetV2.
    - Trains for 5 epochs by default.
    - Saves the trained model to `model.h5`.
- **`app.py`**: The Streamlit interface.
- **`run_app.bat`**: Windows batch script for easy automation.

## 📦 Requirements
- Python 3.8+
- TensorFlow
- Streamlit
- Pandas
- Pillow
- Scipy

---
*Created for the CRCCD Dataset Analysis.*
