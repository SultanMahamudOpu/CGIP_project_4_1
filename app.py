import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IMG_SIZE = 128
MODEL_PATH = 'asl_model_best.h5'
LABELS_PATH = 'class_labels.npy'

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Please train the model first by running: python train_model.py")
        return None, None
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        classes = np.load(LABELS_PATH, allow_pickle=True)
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_sign(model, classes, image):
    """Make prediction on the image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])  # Convert to Python float
    predicted_class = str(classes[predicted_class_idx])  # Convert to Python string
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [(str(classes[idx]), float(predictions[0][idx])) for idx in top_5_idx]
    
    return predicted_class, confidence, top_5_predictions

def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3rem;
            font-size: 18px;
            font-weight: bold;
        }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .info-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        div[data-testid="stFileUploader"] {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px dashed #667eea;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🤟 ASL Sign Language Detector</h1>
            <p style="font-size: 18px; margin: 0;">Upload করুন এবং instant prediction পান!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Model load হচ্ছে..."):
        model, classes = load_model()
    
    if model is None:
        st.stop()
    
    # Success message in a nice card
    st.markdown(f"""
        <div class="info-card">
            <h3 style="margin: 0; color: #28a745;">✅ Model Ready!</h3>
            <p style="margin: 0.5rem 0 0 0;">Trained to recognize <b>{len(classes)} ASL signs</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with better styling
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
            <div class="info-card">
                <p>এই app একটি <b>Convolutional Neural Network (CNN)</b> use করে 
                American Sign Language (ASL) hand signs detect করে।</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 Supported Signs")
        # Display classes in a grid
        cols_per_row = 4
        for i in range(0, len(classes), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(classes):
                    col.markdown(f"<div style='text-align: center; background: #667eea; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem; font-weight: bold;'>{classes[i+j]}</div>", unsafe_allow_html=True)
        
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        <div class="info-card">
            ✅ Clear, well-lit ছবি<br>
            ✅ Hand clearly visible<br>
            ✅ Simple background<br>
            ✅ Sharp & focused image
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center;">
                <p style="color: gray; font-size: 12px;">Made with ❤️ using</p>
                <p style="font-weight: bold;">Streamlit • TensorFlow</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "ASL sign এর ছবি choose করুন",
            type=['jpg', 'jpeg', 'png'],
            help="JPG, JPEG, অথবা PNG format supported",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image with styling
            image = Image.open(uploaded_file)
            st.markdown("""
                <div style="border: 3px solid #667eea; border-radius: 10px; padding: 0.5rem; background: white;">
                </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            # Predict button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Predict Sign", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing করছে..."):
                    try:
                        predicted_class, confidence, top_5 = predict_sign(model, classes, image)
                        
                        # Store results in session state
                        st.session_state.prediction = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.top_5 = top_5
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
        else:
            # Placeholder when no image is uploaded
            st.markdown("""
                <div style="border: 3px dashed #667eea; border-radius: 10px; padding: 3rem; text-align: center; background: #f8f9fa;">
                    <h3 style="color: #667eea;">📸 ছবি upload করুন</h3>
                    <p style="color: gray;">JPG, JPEG অথবা PNG format</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            # Main prediction in a beautiful box
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="margin: 0; font-size: 24px;">🎯 Predicted Sign</h2>
                    <h1 style="font-size: 120px; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {st.session_state.prediction}
                    </h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence with better visualization
            confidence_percent = st.session_state.confidence * 100
            
            # Determine confidence level and color
            if confidence_percent >= 90:
                conf_color = "#28a745"
                conf_label = "🟢 খুবই High Confidence"
                conf_emoji = "🎉"
            elif confidence_percent >= 70:
                conf_color = "#17a2b8"
                conf_label = "🔵 High Confidence"
                conf_emoji = "👍"
            elif confidence_percent >= 50:
                conf_color = "#ffc107"
                conf_label = "🟡 Medium Confidence"
                conf_emoji = "🤔"
            else:
                conf_color = "#dc3545"
                conf_label = "🔴 Low Confidence"
                conf_emoji = "⚠️"
            
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: {conf_color};">{conf_emoji} Confidence: {confidence_percent:.1f}%</h3>
                    <p style="margin: 0.5rem 0 0 0; color: gray;">{conf_label}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(st.session_state.confidence)
            
            # Top 5 predictions with better design
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Top 5 Predictions")
            
            for i, (sign, prob) in enumerate(st.session_state.top_5, 1):
                prob_percent = prob * 100
                
                # Medal emojis for top 3
                medal = ""
                if i == 1:
                    medal = "🥇"
                elif i == 2:
                    medal = "🥈"
                elif i == 3:
                    medal = "🥉"
                else:
                    medal = f"{i}️⃣"
                
                # Create a nice card for each prediction
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 24px;">{medal}</span>
                                <span style="font-size: 24px; font-weight: bold; color: #667eea;">{sign}</span>
                            </div>
                            <span style="font-size: 18px; font-weight: bold; color: #764ba2;">{prob_percent:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(prob, text="")
            
            # Clear button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Clear Results", use_container_width=True):
                for key in ['prediction', 'confidence', 'top_5']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            # Better empty state
            st.markdown("""
                <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 3rem; text-align: center; background: #f8f9fa; margin-top: 2rem;">
                    <h3 style="color: #667eea;">👈 শুরু করুন</h3>
                    <p style="color: gray;">একটি ASL sign এর ছবি upload করুন<br>এবং "Predict Sign" button এ click করুন</p>
                    <p style="font-size: 48px; margin: 1rem 0;">🤟</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
