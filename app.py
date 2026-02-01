import streamlit as st
import os
from PIL import Image
import pandas as pd
import glob
import random
import tensorflow as tf
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Page config
st.set_page_config(page_title="CRCCD Dataset Explorer & Prediction", page_icon="🔬", layout="wide")

st.title("🔬 CRCCD Explorer & Diagnostics")
st.markdown("Explore the dataset and use AI to classify colorectal cancer images.")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")

def get_class_stats(directory):
    """Counts files in each subdirectory."""
    if not os.path.exists(directory):
        return None
    
    stats = {}
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        # Count only images
        images = glob.glob(os.path.join(cls_path, "*.*"))
        # Filter for common image extensions just in case
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        stats[cls] = len(images)
    
    return stats

# --- Sidebar ---
st.sidebar.header("DataSet Control")
dataset_split = st.sidebar.radio("Select Split", ["Train", "Test"])

active_dir = TRAIN_DIR if dataset_split == "Train" else TEST_DIR

if not os.path.exists(active_dir):
    st.error(f"Directory not found: {active_dir}")
    st.stop()

# --- Load Stats ---
stats = get_class_stats(active_dir)

if not stats:
    st.warning("No classes found in this directory.")
    st.stop()

df_stats = pd.DataFrame(list(stats.items()), columns=["Class", "Count"])

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🖼️ Gallery", "🤖 Prediction Analysis", "📈 Advanced Metrics"])

with tab1:
    st.subheader(f"{dataset_split} Dataset Distribution")
    st.bar_chart(df_stats.set_index("Class"))
    
    st.dataframe(df_stats, use_container_width=True)
    
    total_images = df_stats["Count"].sum()
    st.metric("Total Images", total_images)

with tab2:
    st.subheader("Image Gallery")
    
    selected_class = st.selectbox("Select Class", df_stats["Class"])
    
    class_path = os.path.join(active_dir, selected_class)
    all_images = glob.glob(os.path.join(class_path, "*.*"))
    all_images = [img for img in all_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if all_images:
        # Pagination or Random Sample
        sample_size = st.slider("Number of images to show", min_value=1, max_value=min(50, len(all_images)), value=10)
        
        if st.button("Shuffle Images"):
            random.shuffle(all_images)
            
        cols = st.columns(4)
        for i, img_path in enumerate(all_images[:sample_size]):
            with cols[i % 4]:
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=os.path.basename(img_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {os.path.basename(img_path)}")
    else:
        st.info("No images found in this class.")

with tab3:
    st.subheader("AI Diagnosis Assistant")
    
    # Model Selection
    model_choice = st.radio("Select AI Model", ["YOLOv8 (Recommended)", "MobileNetV2 (Legacy)"], horizontal=True)
    
    model_path_mobilenet = os.path.join(BASE_DIR, "model.h5")
    class_names_path = os.path.join(BASE_DIR, "class_names.txt")
    
    # YOLO Model Path
    # It might be in runs/classify/yolo_results/weights/best.pt
    model_path_yolo = os.path.join(BASE_DIR, "yolo_results", "weights", "best.pt")

    model_ready = False
    
    if model_choice == "MobileNetV2 (Legacy)":
        if os.path.exists(model_path_mobilenet) and os.path.exists(class_names_path):
            @st.cache_resource
            def load_mobilenet():
                model = tf.keras.models.load_model(model_path_mobilenet)
                with open(class_names_path, "r") as f:
                    classes = f.read().splitlines()
                return model, classes
            
            with st.spinner("Loading MobileNetV2..."):
                model, class_names = load_mobilenet()
                model_ready = True
        else:
            st.warning("MobileNetV2 model not found.")

    else: # YOLO
        if YOLO is None:
            st.error("Ultralytics library not installed.")
        elif os.path.exists(model_path_yolo):
            @st.cache_resource
            def load_yolo():
                return YOLO(model_path_yolo)
            
            with st.spinner("Loading YOLOv8..."):
                model = load_yolo()
                # Class names are inside the model object
                class_names = list(model.names.values())
                model_ready = True
        else:
            st.warning("YOLOv8 model not found. Please run training.")

    if model_ready:
        st.success(f"{model_choice} loaded successfully!")
        
        # Input selection
        input_method = st.radio("Choose Input Method", ["Upload Image", "Select Random Test Image"])
        
        image_to_predict = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image_to_predict = Image.open(uploaded_file)
        
        else:
            if os.path.exists(TEST_DIR):
                test_classes = os.listdir(TEST_DIR)
                random_class = random.choice(test_classes)
                random_class_path = os.path.join(TEST_DIR, random_class)
                if os.path.exists(random_class_path):
                    images = glob.glob(os.path.join(random_class_path, "*.*"))
                    if images:
                        random_image_path = random.choice(images)
                        image_to_predict = Image.open(random_image_path)
                        st.caption(f"Selected image from: **{random_class}** (Ground Truth)")
            else:
                st.error("Test directory not found.")
                
        if image_to_predict:
            st.image(image_to_predict, caption="Input Image", width=300)
            
            if st.button("Analyze Image"):
                predicted_label = "Unknown"
                confidence = 0.0
                chart_data = None
                
                if model_choice == "MobileNetV2 (Legacy)":
                    try:
                        # Preprocess
                        img = image_to_predict.resize((224, 224))
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0 
                        
                        prediction = model.predict(img_array)
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])
                        predicted_label = class_names[predicted_class_idx]
                        
                        chart_data = pd.DataFrame({
                            "Class": class_names,
                            "Probability": prediction[0]
                        })
                    except Exception as e:
                        st.error(f"Error predicting with MobileNet: {e}")
                
                else: # YOLO
                    try:
                        # Inference
                        # YOLO accepts PIL image directly
                        results = model(image_to_predict)
                        
                        # Extract results
                        # results is a list
                        res = results[0]
                        probs = res.probs.data.cpu().numpy() # Probability array
                        top1_idx = np.argmax(probs)
                        
                        predicted_label = class_names[top1_idx]
                        confidence = probs[top1_idx]
                        
                        chart_data = pd.DataFrame({
                            "Class": class_names,
                            "Probability": probs
                        })
                    except Exception as e:
                        st.error(f"Error predicting with YOLO: {e}")

                st.markdown(f"### Prediction: **{predicted_label}**")
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                
                if chart_data is not None:
                    st.bar_chart(chart_data.set_index("Class"))

with tab4:
    st.subheader("📈 Performance Metrics")
    st.markdown("Metrics generated during YOLOv8 Training.")
    
    # YOLO automatically saves plots in runs/classify/yolo_results
    yolo_results_dir = os.path.join(BASE_DIR, "yolo_results")
    
    if os.path.exists(yolo_results_dir):
        # Confusion Matrix
        cm_path = os.path.join(yolo_results_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
            
        # Results (Loss/Accuracy curves)
        res_path = os.path.join(yolo_results_dir, "results.png")
        if os.path.exists(res_path):
            st.image(res_path, caption="Training Results (Loss & Accuracy)", use_container_width=True)
            
        # Normalized Confusion Matrix
        cm_norm_path = os.path.join(yolo_results_dir, "confusion_matrix_normalized.png")
        if os.path.exists(cm_norm_path):
            st.image(cm_norm_path, caption="Normalized Confusion Matrix", use_container_width=True)
            
    else:
        st.info("No YOLO training results found yet. Train the model to see metrics here.")

st.markdown("---")
st.caption(f"Reading from: {active_dir}")
