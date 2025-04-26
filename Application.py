import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from models import PerformanceModel
import time

# Define emotions
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Define colors for emotions
emotion_colors = {
    "Neutral": (200, 200, 200),
    "Happy": (255, 255, 0),
    "Surprise": (255, 165, 0),
    "Sad": (0, 0, 255),
    "Angry": (255, 0, 0),
    "Disgust": (128, 0, 128),
    "Fear": (0, 255, 255),
    "Contempt": (0, 255, 0)
}

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
    model.eval()
    return model, device

# Load face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model, device = load_model()
face_detector = load_face_detector()

# Performance settings
st.sidebar.title("Performance Settings")
process_every_n_frames = st.sidebar.slider("Process every N frames", 1, 10, 2)
detection_scale = st.sidebar.slider("Detection Scale (%)", 30, 100, 60)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

# Define the video processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_detection_time = time.time()
        self.last_detection_faces = []
        self.last_detection_emotions = []
        self.fps_list = []
        self.last_fps_update = time.time()
        self.current_fps = 0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        start_time = time.time()
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Calculate FPS
        if time.time() - self.last_fps_update > 1.0:  # Update FPS every second
            self.current_fps = len(self.fps_list) / (time.time() - self.last_fps_update)
            self.fps_list = []
            self.last_fps_update = time.time()
        
        self.fps_list.append(time.time())
        
        # Only process every Nth frame for detection
        if self.frame_count % process_every_n_frames == 0:
            # Downscale image for faster processing
            scale_percent = detection_scale
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            small_img = cv2.resize(img, (width, height))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces on smaller image
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            
            # Scale coordinates back to original size
            faces = [(int(x * 100/scale_percent), int(y * 100/scale_percent), 
                     int(w * 100/scale_percent), int(h * 100/scale_percent)) for (x, y, w, h) in faces]
            
            # Process emotions only if faces detected
            self.last_detection_faces = []
            self.last_detection_emotions = []
            
            for (x, y, w, h) in faces:
                face_roi = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                face_img = cv2.resize(face_roi, (48, 48))
                
                # Normalize the image
                face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5)
                face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(device)
                
                # Perform inference
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                    top_idx = np.argmax(probs)
                    emotion = emotions[top_idx]
                
                self.last_detection_faces.append((x, y, w, h))
                self.last_detection_emotions.append(emotion)
            
            self.last_detection_time = time.time()
        
        # Always render the most recent detection results
        for i, (x, y, w, h) in enumerate(self.last_detection_faces):
            if i < len(self.last_detection_emotions):
                emotion = self.last_detection_emotions[i]
                color = emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display FPS if enabled
        if show_fps:
            cv2.putText(img, f"FPS: {self.current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        if processing_time > 0.033:  # If taking more than 33ms (30fps)
            if self.frame_count % 30 == 0:  # Don't spam the log
                print(f"Frame processing took {processing_time*1000:.1f}ms (slow)")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-Time Facial Emotion Recognition")
st.markdown("""
This app detects emotions in real-time using your webcam.
Adjust the performance settings in the sidebar if you experience lag.
""")

webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # Process frames asynchronously
)