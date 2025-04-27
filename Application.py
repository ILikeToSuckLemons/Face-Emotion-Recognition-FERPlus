import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from models import PerformanceModel
from emotionoverlay import EmotionOverlay
from gifoverlay import GifEmotionOverlay
import pandas as pd
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Real-time Facial Emotion Recognition", layout="wide")
st.title("Real-time Facial Emotion Recognition")

# Initialize session state for emotion data storage
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []
    
if 'show_graphs' not in st.session_state:
    st.session_state.show_graphs = False

# Load model only once
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model/ferplus_model_pd_acc.pth"
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Global variables
model, device = load_model()
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
gif_overlay = GifEmotionOverlay("EmojiGif/")

# Emotion color definitions
emotion_colors = {
    "Neutral": (255, 255, 255),  # White
    "Happy": (0, 255, 255),      # Yellow
    "Surprise": (0, 165, 255),   # Orange
    "Sad": (255, 0, 0),          # Blue
    "Angry": (0, 0, 255),        # Red
    "Disgust": (128, 0, 128),    # Purple
    "Fear": (255, 255, 0),       # Cyan
    "Contempt": (0, 255, 0)      # Green
}

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.faces = []
        self.start_time = time.time()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Downscale image for processing
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (int(w * resolution_factor), int(h * resolution_factor)))
        scale_factor = 1 / resolution_factor
        
        # Process frame
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Run face detection less frequently for better performance
        if self.frame_count % detect_frequency == 0:
            self.faces = haar_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5, 
                minSize=(int(48 * resolution_factor), int(48 * resolution_factor))
            )
            
        self.frame_count += 1
        
        # Scale back face coordinates to original image size
        scaled_faces = [(int(x * scale_factor), int(y * scale_factor), 
                        int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in self.faces]
        
        # Store emotion data for this frame
        current_time = time.time()
        frame_emotions = {emotion: 0 for emotion in emotions}
        
        for (x, y, w, h) in scaled_faces:
            face_region = img[y:y+h, x:x+w]
            if face_region.size == 0:
                continue
                
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_tensor = torch.tensor(face_resized, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
            
            # Run model inference
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_emotion_idx = np.argmax(probs)
                top_emotion = emotions[top_emotion_idx]
            
            # Record emotion data for all faces
            for i, prob in enumerate(probs):
                frame_emotions[emotions[i]] += prob
            
            # Overlay character and draw rectangle
            img = gif_overlay.overlay_gif(img, top_emotion, x, y, w, h, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), emotion_colors[top_emotion], 2)
            
            # Display emotion text
            text = f"{top_emotion}: {int(probs[top_emotion_idx] * 100)}%"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        # Store emotion data if faces were detected
        if len(scaled_faces) > 0:
            avg_emotions = {k: v/len(scaled_faces) if len(scaled_faces) > 0 else 0 for k, v in frame_emotions.items()}
            
            # Append to session state directly
            st.session_state.emotion_data.append({
                'timestamp': current_time - self.start_time,
                **avg_emotions
            })
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def display_emotion_graphs():
    if not st.session_state.emotion_data:
        st.warning("No emotion data recorded. Start the webcam and show your face to record emotions.")
        return
    
    df = pd.DataFrame(st.session_state.emotion_data)
    
    st.subheader("Emotion Analysis")
    
    # Create two columns for the graphs
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of average emotions
        st.write("### Average Emotion Distribution")
        fig1, ax1 = plt.subplots()
        avg_emotions = df[emotions].mean().sort_values(ascending=False)
        colors = [emotion_colors[emotion] for emotion in avg_emotions.index]
        bars = ax1.bar(avg_emotions.index, avg_emotions.values * 100, color=colors)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_ylim(0, 100)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col2:
        # Line chart of emotions over time
        st.write("### Emotions Over Time")
        fig2, ax2 = plt.subplots()
        for emotion in emotions:
            ax2.plot(df['timestamp'], df[emotion] * 100, label=emotion, 
                    color=[c/255 for c in emotion_colors[emotion]])
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig2)

# Sidebar settings
st.sidebar.title("Settings")
detect_frequency = st.sidebar.slider("Face Detection Frequency", 1, 10, 5)
resolution_factor = st.sidebar.slider("Resolution Scale", 0.5, 1.0, 0.75, 0.05)

# Main app
webrtc_ctx = webrtc_streamer(
    key="emotion-recognition",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 30},
        "audio": False
    },
    async_processing=True,
)

# Always show the graphs if we have data
if st.session_state.emotion_data:
    display_emotion_graphs()
    if st.button("Clear Data"):
        st.session_state.emotion_data = []
        st.experimental_rerun()

# Add information section
with st.expander("About this app"):
    st.write("""
    This app performs real-time facial emotion recognition using deep learning.
    - The webcam feed analyzes your facial expressions
    - Emotion data is collected while the webcam is active
    - Graphs show your emotional patterns automatically
    - Use the sidebar to adjust performance settings
    """)