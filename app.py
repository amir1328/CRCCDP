import streamlit as st
import os
from PIL import Image
import pandas as pd
import glob
import random
import tensorflow as tf
import numpy as np

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
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🖼️ Gallery", "🤖 Prediction Analysis"])

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
    
    model_path = os.path.join(BASE_DIR, "model.h5")
    class_names_path = os.path.join(BASE_DIR, "class_names.txt")
    
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        @st.cache_resource
        def load_model_and_classes():
            model = tf.keras.models.load_model(model_path)
            with open(class_names_path, "r") as f:
                classes = f.read().splitlines()
            return model, classes

        with st.spinner("Loading AI Model..."):
            model, class_names = load_model_and_classes()
            
        st.success("Model loaded successfully!")
        
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
                # Preprocess
                img = image_to_predict.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Rescale match training
                
                prediction = model.predict(img_array)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                predicted_label = class_names[predicted_class_idx]
                
                st.markdown(f"### Prediction: **{predicted_label}**")
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                
                # Bar chart of probabilities
                chart_data = pd.DataFrame({
                    "Class": class_names,
                    "Probability": prediction[0]
                })
                st.bar_chart(chart_data.set_index("Class"))
                
    else:
        st.warning("Model (`model.h5`) not found. Please run the training script or restart the app application via `run_app.bat`.")

st.markdown("---")
st.caption(f"Reading from: {active_dir}")
