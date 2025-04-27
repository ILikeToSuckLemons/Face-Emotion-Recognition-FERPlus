import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import deque

# Initialize session state more robustly
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = deque(maxlen=1000)  # Using deque with maxlen to prevent memory issues
if 'show_graphs' not in st.session_state:
    st.session_state.show_graphs = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

# Set page config
st.set_page_config(page_title="Real-time Facial Emotion Recognition", layout="wide")
st.title("Real-time Facial Emotion Recognition")

# Load model (simplified for example - use your actual model loading code)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model/ferplus_model_pd_acc.pth"
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion colors (keeping your original styling)
emotion_colors = {
    "Neutral": (255, 255, 255),
    "Happy": (0, 255, 255),
    "Surprise": (0, 165, 255),
    "Sad": (255, 0, 0),
    "Angry": (0, 0, 255),
    "Disgust": (128, 0, 128),
    "Fear": (255, 255, 0),
    "Contempt": (0, 255, 0)
}

emotion_text_colors = {
    "Neutral": [(255,255,255), (224,212,196), (228,203,179)],
    "Happy": [(182,110,68), (76,235,253), (83,169,242)],
    "Surprise": [(247,255,0), (42,42,165), (232,206,0)],
    "Sad": [(194,105,3), (228,172,32), (237,202,162)],
    "Angry": [(61, 57, 242), (49,121,249), (232,220,214)],
    "Disgust": [(70,190,77), (120,159,6), (100,55,124)],
    "Fear": [(198, 128, 134), (133,71,68), (80,45,98)],
    "Contempt": [(160, 134, 72), (145, 180, 250), (173, 217, 251)]
}

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.color_index = 0
        self.animation_offset = 0
        self.offset_direction = 1
        self.faces = []
        self.start_time = time.time()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Face detection and processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.faces = haar_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Color cycling animation
        if self.frame_count % 10 == 0:
            self.color_index = (self.color_index + 1) % 3
            
        # Process each face
        frame_emotions = {e: 0 for e in emotions}
        for (x, y, w, h) in self.faces:
            face_region = gray[y:y+h, x:x+w]
            if face_region.size == 0:
                continue
                
            # Prepare face for model
            face_resized = cv2.resize(face_region, (48, 48))
            face_tensor = torch.tensor(face_resized, dtype=torch.float32).div(255).sub(0.5).div(0.5)
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_emotion_idx = np.argmax(probs)
                top_emotion = emotions[top_emotion_idx]
            
            # Update frame emotions
            for i, e in enumerate(emotions):
                frame_emotions[e] += probs[i]
            
            # Visualization (your original code)
            img = gif_overlay.overlay_gif(img, top_emotion, x, y, w, h, self.animation_offset)
            cv2.rectangle(img, (x, y), (x+w, y+h), emotion_colors[top_emotion], 2)
            
            # Show all emotions as text
            for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                if i == top_emotion_idx:
                    text_color = emotion_text_colors[top_emotion][self.color_index]
                else:
                    text_color = (255, 255, 255)
                text = f"{emotion}: {int(prob*100)}%"
                cv2.putText(img, text, (x, y-10-(i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Update emotion data (critical fix - using deque)
        if len(self.faces) > 0:
            avg_emotions = {k: v/len(self.faces) for k, v in frame_emotions.items()}
            st.session_state.emotion_data.append({
                'timestamp': current_time - self.start_time,
                **avg_emotions
            })
            
            # Limit update rate to prevent performance issues
            if current_time - st.session_state.last_update > 1.0:  # Update once per second max
                st.session_state.last_update = current_time
                st.experimental_rerun()
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def display_emotion_graphs():
    if not st.session_state.emotion_data:
        st.warning("No emotion data collected yet. Show your face to the webcam.")
        return
    
    df = pd.DataFrame(st.session_state.emotion_data)
    
    # Create two columns for graphs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emotion Distribution")
        fig1, ax1 = plt.subplots(figsize=(8,5))
        avg_emotions = df[emotions].mean().sort_values(ascending=False)
        colors = [tuple(c/255 for c in emotion_colors[e]) for e in avg_emotions.index]
        avg_emotions.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_ylabel("Probability (%)")
        ax1.set_ylim(0, 100)
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Emotion Over Time")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for e in emotions:
            ax2.plot(df['timestamp'], df[e]*100, label=e, 
                    color=tuple(c/255 for c in emotion_colors[e]))
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Probability (%)")
        ax2.legend(bbox_to_anchor=(1.05, 1))
        ax2.set_ylim(0, 100)
        st.pyplot(fig2)

# Main app layout
col1, col2 = st.columns([3, 1])

with col1:
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.header("Emotion Data")
    if st.button("Show Graphs"):
        st.session_state.show_graphs = True
    
    if st.button("Clear Data"):
        st.session_state.emotion_data.clear()
        st.session_state.show_graphs = False
        st.experimental_rerun()

# Always show graphs if we have data and button was pressed
if st.session_state.show_graphs and st.session_state.emotion_data:
    display_emotion_graphs()

# Information section
with st.expander("How to use this app"):
    st.write("""
    1. Allow camera access when prompted
    2. Show your face to the camera
    3. Emotions will be detected and displayed
    4. Click 'Show Graphs' to view your emotion patterns
    5. Use 'Clear Data' to reset the analysis
    """)