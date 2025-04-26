import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Import your custom model
from models import PerformanceModel

# App title and description
st.title("Real-time Facial Emotion Recognition")
st.write("Allow camera access to detect emotions in real-time!")

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
        # Use LBP cascade for faster detection
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

# Generate emotion probability chart
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

# Video processor class for webcam
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, model, device, face_detector):
        self.model = model
        self.device = device
        self.face_detector = face_detector
        self.last_emotion = None
        self.last_probs = None
        self.frame_count = 0
        self.last_process_time = time.time()
        self.process_every_n_frames = 3  # Process every 3rd frame to reduce lag
        
    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Only process every nth frame to reduce lag
        if self.frame_count % self.process_every_n_frames == 0:
            current_time = time.time()
            process_time = current_time - self.last_process_time
            self.last_process_time = current_time
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces - use scaleFactor to improve performance
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.2,  # Increase for better performance
                minNeighbors=4,   # Lower for performance
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+h]
                
                # Resize to model input size
                face_img = cv2.resize(face_roi, (48, 48))
                
                # Predict emotion
                probs, emotion, emotion_idx = predict_emotion(face_img, self.model, self.device)
                
                if emotion:
                    # Save last emotion and probabilities for display
                    self.last_emotion = emotion
                    self.last_probs = probs
                    
                    # Draw rectangle around face - directly on BGR image
                    color_bgr = emotion_colors.get(emotion, (255, 255, 255))
                    # Convert RGB to BGR for OpenCV
                    color_bgr = (color_bgr[2], color_bgr[1], color_bgr[0])
                    cv2.rectangle(img, (x, y), (x+w, y+h), color_bgr, 2)
                    
                    # Draw emotion label
                    text = f"{emotion}"
                    cv2.putText(img, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
        
        # Return the frame directly without extra conversion
        return frame.from_ndarray(img)

# Main app
def main():
    # Add sidebar settings
    st.sidebar.title("Performance Settings")
    processing_rate = st.sidebar.slider("Processing rate", 1, 10, 3, 
                                     help="Higher values = better performance but lower detection rate")
    
    # Load model and face detector
    model, device = load_model()
    face_detector = load_face_detector()
    
    if model is None or face_detector is None:
        st.error("Failed to load required components. Please check the error messages above.")
        return
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Real-time Webcam", "Upload Image"])
    
    with tab1:
        st.header("Real-time Emotion Detection")
        
        # Create WebRTC streamer with STUN servers for Streamlit Cloud
        rtc_config = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]}
        )
        
        # Create video processor
        processor = EmotionVideoProcessor(model, device, face_detector)
        processor.process_every_n_frames = processing_rate  # Apply user setting
        
        # Create WebRTC streamer with improved settings
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=lambda: processor,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,  # Process frames asynchronously
        )
        
        # Create placeholder for emotion chart
        chart_placeholder = st.empty()
        
        # Update chart when streaming is active
        if webrtc_ctx.state.playing and processor.last_probs is not None:
            chart = generate_emotion_chart(processor.last_probs)
            chart_placeholder.pyplot(chart)
            
            # Display detected emotion
            if processor.last_emotion:
                st.info(f"Current emotion: {processor.last_emotion}")
    
    with tab2:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image from uploaded file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Convert to RGB for display and grayscale for detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
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
            
            # Display the image with emotion detection
            st.image(img_rgb, caption="Detected Emotions", use_column_width=True)
            
            # Check if faces were detected
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

if __name__ == "__main__":
    main()