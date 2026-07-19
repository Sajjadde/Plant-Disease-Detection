# app.py - Final Production Version
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Page Configuration
st.set_page_config(
    page_title="🌿 Plant Disease Detection System",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection System")
st.markdown("---")

# ============================================================
# Section 1: Check Required Files
# ============================================================
required_files = ['class_names.json', 'model_info.json']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files: {', '.join(missing_files)}")
    st.info("ℹ️ Please run train.py first to build the model.")
    st.stop()

# ============================================================
# Section 2: Load Model with Multiple Fallback Methods
# ============================================================
@st.cache_resource
def load_model_with_fallback():
    """Load the model using multiple fallback methods for robustness"""
    
    # Method 1: Load from SavedModel folder using tf.keras.models.load_model
    try:
        if os.path.exists('saved_model'):
            st.info("🔄 Loading model from SavedModel with tf.keras.models.load_model...")
            model = tf.keras.models.load_model('saved_model')
            st.success("✅ Model loaded from SavedModel!")
            return model
    except Exception as e:
        st.warning(f"⚠️ SavedModel method failed: {str(e)[:100]}...")
    
    # Method 2: Load from .keras format
    try:
        if os.path.exists('plant_disease_model.keras'):
            st.info("🔄 Loading model from .keras format...")
            model = tf.keras.models.load_model('plant_disease_model.keras')
            st.success("✅ Model loaded from .keras format!")
            return model
    except Exception as e:
        st.warning(f"⚠️ .keras method failed: {str(e)[:100]}...")
    
    # Method 3: Load from .h5 format
    try:
        if os.path.exists('plant_disease_model.h5'):
            st.info("🔄 Loading model from .h5 format...")
            model = tf.keras.models.load_model('plant_disease_model.h5')
            st.success("✅ Model loaded from .h5 format!")
            return model
    except Exception as e:
        st.warning(f"⚠️ .h5 method failed: {str(e)[:100]}...")
    
    # Method 4: Load from architecture and weights
    try:
        if os.path.exists('model_architecture.json') and os.path.exists('model_weights.h5'):
            st.info("🔄 Loading model from architecture and weights...")
            with open('model_architecture.json', 'r') as f:
                model_json = json.load(f)
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('model_weights.h5')
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            st.success("✅ Model loaded from architecture and weights!")
            return model
    except Exception as e:
        st.warning(f"⚠️ Architecture/weights method failed: {str(e)[:100]}...")
    
    # If all methods fail
    st.error("❌ No method succeeded in loading the model!")
    return None

# ============================================================
# Section 3: Load Model Information
# ============================================================
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        try:
            with open('model_info.json', 'r', encoding='utf-8') as f:
                info = json.load(f)
                return info['class_names']
        except:
            return None

@st.cache_data
def load_model_info():
    try:
        with open('model_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {'model_type': 'MobileNetV2', 'input_shape': (128, 128, 3)}

# ============================================================
# Section 4: Execute Loading Process
# ============================================================
with st.spinner("🔄 Loading model..."):
    model = load_model_with_fallback()
    class_names = load_class_names()
    model_info = load_model_info()

if model is None:
    st.error("❌ Failed to load the model!")
    st.stop()

if class_names is None:
    st.error("❌ Failed to load class names!")
    st.stop()

st.success(f"✅ Model loaded successfully! Number of classes: {len(class_names)}")

# Display model information
with st.expander("📊 Model Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", model_info.get('model_type', 'MobileNetV2'))
    with col2:
        input_shape = model_info.get('input_shape', (128, 128, 3))
        st.metric("Input Size", f"{input_shape[0]}×{input_shape[1]}")
    with col3:
        st.metric("Number of Classes", len(class_names))

# ============================================================
# Section 5: Image Upload and Prediction
# ============================================================
st.markdown("---")
st.subheader("📤 Upload Image")

col_left, col_right = st.columns(2)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

def preprocess_image(image):
    """Preprocess image for model input"""
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def clean_class_name(name):
    """Clean class name by removing path and underscores"""
    if '/' in name:
        name = name.split('/')[-1]
    if '\\' in name:
        name = name.split('\\')[-1]
    name = name.replace('_', ' ')
    # Remove duplicate words
    parts = name.split(' - ')
    if len(parts) > 1:
        name = parts[-1]
    return name

if uploaded_file is not None:
    with col_left:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
        
        # Display file details
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        st.json(file_details)
    
    with col_right:
        if st.button("🔍 Diagnose Disease", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing image..."):
                try:
                    # Open and preprocess the image
                    image = Image.open(uploaded_file)
                    img_array = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    # Display results
                    st.subheader("🎯 Diagnosis Result")
                    
                    predicted_class = clean_class_name(class_names[predicted_class_idx])
                    
                    # Display with color coding based on confidence level
                    if confidence > 0.8:
                        st.success(f"### ✅ **{predicted_class}**")
                        st.info(f"✨ Confidence: {confidence*100:.2f}%")
                    elif confidence > 0.6:
                        st.warning(f"### ⚠️ **{predicted_class}**")
                        st.info(f"✨ Confidence: {confidence*100:.2f}%")
                    else:
                        st.error(f"### ❌ **Low Confidence Diagnosis**")
                        st.info(f"✨ Confidence: {confidence*100:.2f}%")
                    
                    # Progress bar
                    st.progress(float(confidence))
                    
                    # Display top 3 classes
                    st.subheader("📊 Top 3 Probabilities")
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    
                    for idx in top_3_idx:
                        class_name = clean_class_name(class_names[idx])
                        prob = predictions[0][idx] * 100
                        # Display with mini progress bar
                        st.write(f"**{class_name}**")
                        st.progress(float(predictions[0][idx]))
                        st.caption(f"Probability: {prob:.1f}%")
                        st.divider()
                        
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.error("Please try again or contact support.")

# ============================================================
# Section 6: Sidebar - Help and Additional Information
# ============================================================
with st.sidebar:
    st.header("📖 Help Guide")
    st.markdown("""
    ### How to Use:
    1. Click on **Browse files** button
    2. Select a leaf image
    3. Click on **Diagnose Disease** button
    4. View the results
    
    ---
    
    ### 🔬 About the Model:
    - **Architecture:** MobileNetV2
    - **Input Size:** 128×128 pixels
    - **Number of Classes:** {}
    - **Framework:** TensorFlow Keras
    
    ---
    
    ### 📌 Tips:
    - Use well-lit images for best results
    - Ensure the leaf is clearly visible in the image
    - Simple backgrounds help with accurate diagnosis
    """.format(len(class_names)))
    
    # Display list of classes
    with st.expander("📋 Class List"):
        st.write(f"**Total Classes:** {len(class_names)}")
        
        # Display classes in two columns
        cols = st.columns(2)
        for i, name in enumerate(class_names):
            clean_name = clean_class_name(name)
            cols[i % 2].write(f"• {clean_name}")
        
        st.caption(f"Showing {len(class_names)} classes")

# ============================================================
# Section 7: Footer
# ============================================================
st.markdown("---")
st.caption("🌿 Built with ❤️ using TensorFlow and Streamlit")