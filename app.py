# app.py
import streamlit as st
from PIL import Image
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# Set page config
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")
st.title("🫁 Pneumonia Detection from Chest X-Rays")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = models.load_model("pneumonia_model.h5")
        return model
    except FileNotFoundError:
        st.warning("Model file 'pneumonia_model.h5' not found. Please train and save the model first.")
        return None

# Paths
data_dir = "Lung X-Ray Image"
normal_path = os.path.join(data_dir, "Normal")
viral_path = os.path.join(data_dir, "Viral Pneumonia")

normal_gray_path = os.path.join(data_dir, "Normal_Gray")
viral_gray_path = os.path.join(data_dir, "Viral Pneumonia_Gray")

input_dirs = {"Normal": normal_path, "Viral Pneumonia": viral_path}
output_dirs = {"Normal": normal_gray_path, "Viral Pneumonia": viral_gray_path}

# Create preprocessed directories if they don't exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Function to preprocess to grayscale
def preprocess_to_gray(image_path, save_path):
    img = Image.open(image_path).convert("L")
    img.save(save_path)
    return img

# Preprocess all images
for label, input_dir in input_dirs.items():
    if os.path.exists(input_dir):
        output_dir = output_dirs[label]
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if os.path.isfile(input_path) and not os.path.exists(output_path):
                try:
                    preprocess_to_gray(input_path, output_path)
                except Exception as e:
                    st.warning(f"Could not process {filename}: {e}")

# Prediction function
def predict_image(image, model):
    """Predict whether X-Ray is Normal or Viral Pneumonia"""
    img_array = np.array(image.resize((150, 150))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]
    
    if confidence < 0.5:
        label = "Normal"
        confidence_pct = (1 - confidence) * 100
    else:
        label = "Viral Pneumonia"
        confidence_pct = confidence * 100
    
    return label, confidence_pct

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["View Dataset", "Make Prediction", "Compare Images"])

# Tab 1: View Dataset
with tab1:
    st.header("Dataset Viewer")
    st.markdown("View sample X-Ray images from the dataset")
    
    selected_class = st.selectbox("Select Class", ["Normal", "Viral Pneumonia"])
    
    if os.path.exists(input_dirs[selected_class]):
        image_files = os.listdir(input_dirs[selected_class])
        if len(image_files) > 5:
            image_files = random.sample(image_files, 5)
        
        st.subheader(f"{selected_class} X-Rays (Sample)")
        for filename in image_files:
            try:
                orig_path = os.path.join(input_dirs[selected_class], filename)
                orig_img = Image.open(orig_path).convert("RGB")
                st.image(orig_img, caption=filename, width=200)
            except Exception as e:
                st.error(f"Could not load {filename}: {e}")
    else:
        st.error(f"Directory not found: {input_dirs[selected_class]}")

# Tab 2: Make Prediction
with tab2:
    st.header("Predict X-Ray Classification")
    st.markdown("Upload an X-Ray image or select from dataset to classify")
    
    model = load_model()
    
    if model is not None:
        prediction_type = st.radio("Choose input method:", ["Upload Image", "Select from Dataset"])
        
        if prediction_type == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", width=250)
                
                with col2:
                    if st.button("🔍 Predict"):
                        label, confidence = predict_image(image, model)
                        st.success(f"**Prediction: {label}**")
                        st.info(f"**Confidence: {confidence:.2f}%**")
        else:
            selected_class = st.selectbox("Select Class", ["Normal", "Viral Pneumonia"], key="dataset_select")
            
            if os.path.exists(input_dirs[selected_class]):
                image_files = os.listdir(input_dirs[selected_class])
                selected_image = st.selectbox("Select an image", image_files)
                
                if selected_image:
                    image_path = os.path.join(input_dirs[selected_class], selected_image)
                    image = Image.open(image_path).convert("RGB")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption=selected_image, width=250)
                    
                    with col2:
                        if st.button("🔍 Predict"):
                            label, confidence = predict_image(image, model)
                            st.success(f"**Prediction: {label}**")
                            st.info(f"**Confidence: {confidence:.2f}%**")

# Tab 3: Compare Original vs Grayscale
with tab3:
    st.header("Original vs Processed Images")
    st.markdown("Compare original RGB X-Rays with processed grayscale versions")
    
    selected_class = st.selectbox("Select Class", ["Normal", "Viral Pneumonia"], key="compare_select")
    
    if os.path.exists(input_dirs[selected_class]):
        image_files = os.listdir(input_dirs[selected_class])
        if len(image_files) > 5:
            image_files = random.sample(image_files, 5)
        
        st.subheader(f"{selected_class} X-Rays - Original vs Grayscale")
        for filename in image_files:
            try:
                col1, col2 = st.columns(2)
                orig_path = os.path.join(input_dirs[selected_class], filename)
                gray_path = os.path.join(output_dirs[selected_class], filename)
                
                orig_img = Image.open(orig_path).convert("RGB")
                gray_img = Image.open(gray_path)
                
                with col1:
                    st.image(orig_img, caption="Original RGB", width=220)
                with col2:
                    st.image(gray_img, caption="Processed Grayscale", width=220)
            except Exception as e:
                st.error(f"Could not load {filename}: {e}")
    else:
        st.error(f"Directory not found: {input_dirs[selected_class]}")
