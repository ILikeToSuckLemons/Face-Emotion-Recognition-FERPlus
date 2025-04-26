import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import time
import threading
import io

# Import your custom model
from models import PerformanceModel

# App title and description
st.title("Facial Emotion Recognition")
st.write("Use webcam or upload an image to detect emotions!")

# Define emotions list
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Define emotion colors for visualization
emotion_colors = {
    "Neutral": (200, 200, 200),  # Light gray
    "Happy": (255, 255, 0),      # Yellow
    "Surprise": (255, 165, 0),   # Orange
    "Sad": (0, 0, 255),          # Blue
    "Angry": (255, 0, 0),        # Red
    "Disgust": (128, 0, 128),    # Purple
    "Fear": (0, 255, 255),       # Cyan
    "Contempt": (0, 255, 0)      # Green
}

# Load model function
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
        model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Get face detector
@st.cache_resource
def load_face_detector():
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None

# Emotion prediction function
def predict_emotion(face_img, model, device):
    try:
        # Preprocess face
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]
        
        return probs, top_emotion, top_emotion_idx
    except Exception as e:
        st.error(f"Error predicting emotion: {str(e)}")
        return None, None, None

# Process image function
def process_image(img, model, device, face_detector):
    try:
        # Convert to RGB for display and grayscale for detection
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            gray = img
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+h]
            
            # Resize to model input size
            face_img = cv2.resize(face_roi, (48, 48))
            
            # Predict emotion
            probs, emotion, emotion_idx = predict_emotion(face_img, model, device)
            
            if emotion:
                # Draw rectangle around face
                color = emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                text = f"{emotion}"
                cv2.putText(img_rgb, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Add to results
                results.append((x, y, w, h, emotion, probs))
        
        return img_rgb, results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, []

# Function to generate emotion probability chart for a face
def generate_emotion_chart(probs):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [emotion_colors.get(emotion, (255, 255, 255)) for emotion in emotions]
    # Convert RGB to matplotlib format (0-1)
    colors_norm = [(r/255, g/255, b/255) for (r, g, b) in colors]
    
    bars = ax.bar(emotions, probs * 100, color=colors_norm)
    ax.set_ylabel('Probability %')
    ax.set_title('Emotion Probabilities')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(probs):
        ax.text(i, v*100 + 1, f"{v*100:.1f}%", ha='center')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Load model and face detector
    model, device = load_model()
    face_detector = load_face_detector()
    
    if model is None or face_detector is None:
        st.error("Failed to load required components. Please check the error messages above.")
        return
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Camera", "Upload Image"])
    
    with tab1:
        st.header("Camera Emotion Detection")
        
        # Create a camera capture section
        st.write("Click 'Start Camera' to begin emotion detection.")
        
        # Create placeholders
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        emotion_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Session state for camera control
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
            
        if 'current_emotion' not in st.session_state:
            st.session_state.current_emotion = None
            
        if 'current_probs' not in st.session_state:
            st.session_state.current_probs = None
            
        # Handle start button
        if start_button:
            st.session_state.camera_running = True
            
        # Handle stop button
        if stop_button:
            st.session_state.camera_running = False
            
        # Camera capture and processing
        if st.session_state.camera_running:
            status_placeholder.info("Camera is starting... If no video appears, check your browser permissions.")
            
            # Use streamlit's camera_input for simplicity
            camera_image = st.camera_input("Take a picture for emotion detection", key="emotion_camera")
            
            if camera_image is not None:
                # Convert camera input to OpenCV format
                # Convert camera input to RGB format correctly
                

                bytes_data = camera_image.getvalue()
                img_pil = Image.open(io.BytesIO(bytes_data))
                img = np.array(img_pil)  # Now img is in RGB format

                
                # Process the image
                img_rgb, results = process_image(img, model, device, face_detector)
                
                if img_rgb is not None:
                    # Display processed image
                    frame_placeholder.image(img_rgb, caption="Detected Emotions", use_column_width=True)
                    
                    # Display results
                    if len(results) == 0:
                        status_placeholder.warning("No faces detected in the image.")
                    else:
                        status_placeholder.success(f"Detected {len(results)} face(s) in the image.")
                        
                        # Store and display the first face's emotion
                        if results:
                            st.session_state.current_emotion = results[0][4]
                            st.session_state.current_probs = results[0][5]
                            
                            emotion_placeholder.info(f"Current emotion: {st.session_state.current_emotion}")
                            
                            # Create and display chart
                            fig = generate_emotion_chart(st.session_state.current_probs)
                            chart_placeholder.pyplot(fig)
        else:
            status_placeholder.info("Camera is stopped. Click 'Start Camera' to begin.")
    
    with tab2:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image from uploaded file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process the image
            img_rgb, results = process_image(img, model, device, face_detector)
            
            if img_rgb is not None:
                # Display processed image
                st.image(img_rgb, caption="Detected Emotions", use_column_width=True)
                
                # Display results
                if len(results) == 0:
                    st.warning("No faces detected in the image!")
                else:
                    st.success(f"Detected {len(results)} face(s) in the image!")
                    
                    # Display emotion probabilities for each face
                    for i, (_, _, _, _, emotion, probs) in enumerate(results):
                        st.subheader(f"Face #{i+1} - {emotion}")
                        
                        # Create and display chart
                        fig = generate_emotion_chart(probs)
                        st.pyplot(fig)

    # Setup for snapshot mode
    st.sidebar.title("Settings")
    st.sidebar.markdown("### Snapshot Mode")
    
    # Checkbox to enable auto-refresh snapshots
    if st.sidebar.checkbox("Enable auto snapshots (experimental)", value=False):
        interval = st.sidebar.slider("Snapshot interval (seconds)", 2, 10, 3)
        st.sidebar.info(f"Page will automatically capture snapshots every {interval} seconds")
        
        # Add auto-refresh JavaScript
        st.markdown(
            f"""
            <script>
                function takeSnapshot() {{
                    const captureButton = document.querySelector('.stCamera button');
                    if (captureButton) {{
                        captureButton.click();
                    }}
                }}
                
                // Set interval for auto snapshots
                const interval = setInterval(takeSnapshot, {interval * 1000});
                
                // Cleanup on page change/reload
                window.addEventListener('beforeunload', function() {{
                    clearInterval(interval);
                }});
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()