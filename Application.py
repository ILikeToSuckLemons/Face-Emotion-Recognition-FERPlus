import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from models import PerformanceModel
import time
from collections import deque

# Force CPU and optimize for it
torch.set_num_threads(2)  # Limit CPU threads
device = torch.device("cpu")

st.set_page_config(page_title="CPU-Optimized Emotion Recognition", layout="wide")
st.title("CPU-Optimized Emotion Recognition")

@st.cache_resource
def load_lightweight_model():
    model_path = "model/ferplus_model_pd_acc.pth"
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Optimize for CPU inference
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 48, 48)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.eval()
    
    return traced_model

model = load_lightweight_model()
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Ultra-aggressive settings for CPU
INFERENCE_EVERY_N_FRAMES = 45  # Process emotion every 45 frames (1.5 seconds at 30fps)
FACE_DETECTION_EVERY_N_FRAMES = 60  # Detect faces every 2 seconds
PROCESSING_SCALE = 0.25  # Process at 25% resolution
MAX_FACES = 1  # Only process 1 face

# Simple emotion colors
emotion_colors = {
    "Neutral": (128, 128, 128), "Happy": (0, 255, 255), "Surprise": (0, 165, 255),
    "Sad": (255, 0, 0), "Angry": (0, 0, 255), "Disgust": (128, 0, 128),
    "Fear": (255, 255, 0), "Contempt": (0, 255, 0)
}

class CPUOptimizedProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.faces = []
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.last_update_time = time.time()
        self.fps_history = deque(maxlen=10)
        
        # Pre-allocate everything possible
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_tensor = torch.zeros((1, 1, 48, 48), dtype=torch.float32)
        
    def recv(self, frame):
        start_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        
        # Ultra-aggressive downscaling
        h, w = img.shape[:2]
        small_w, small_h = int(w * PROCESSING_SCALE), int(h * PROCESSING_SCALE)
        img_small = cv2.resize(img, (small_w, small_h))
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Face detection (very infrequent)
        if self.frame_count % FACE_DETECTION_EVERY_N_FRAMES == 0:
            faces = self.haar_cascade.detectMultiScale(
                gray_small, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20)
            )
            if len(faces) > 0:
                # Take largest face only
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                # Scale back to original coordinates
                scale_factor = 1 / PROCESSING_SCALE
                self.faces = [(int(largest_face[0] * scale_factor), 
                              int(largest_face[1] * scale_factor),
                              int(largest_face[2] * scale_factor), 
                              int(largest_face[3] * scale_factor))]
        
        # Emotion inference (extremely infrequent)
        if self.frame_count % INFERENCE_EVERY_N_FRAMES == 0 and len(self.faces) > 0:
            x, y, w, h = self.faces[0]
            
            # Quick bounds check
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                y_end = min(y + h, img.shape[0])
                x_end = min(x + w, img.shape[1])
                face_region = img[y:y_end, x:x_end]
                
                if face_region.size > 0:
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    face_48 = cv2.resize(face_gray, (48, 48))
                    
                    # Reuse tensor
                    self.face_tensor[0, 0] = torch.from_numpy(face_48).float().div(255).sub(0.5).div(0.5)
                    
                    with torch.no_grad():
                        outputs = model(self.face_tensor)
                        probs = F.softmax(outputs, dim=1)[0].numpy()
                        
                    top_idx = np.argmax(probs)
                    self.current_emotion = emotions[top_idx]
                    self.emission_confidence = float(probs[top_idx])
        
        # Simple rendering
        if len(self.faces) > 0:
            x, y, w, h = self.faces[0]
            color = emotion_colors.get(self.current_emotion, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion text
            text = f"{self.current_emotion}"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS calculation
        frame_time = time.time() - start_time
        self.fps_history.append(frame_time)
        if len(self.fps_history) > 1:
            avg_fps = 1.0 / np.mean(self.fps_history)
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Minimal WebRTC setup
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="cpu-emotion",
    video_processor_factory=CPUOptimizedProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"width": 320, "height": 240, "frameRate": 15},  # Very low resolution
        "audio": False
    },
    async_processing=True,
)

st.warning("⚠️ This is a CPU-optimized version. Emotions update every ~1.5 seconds to maintain performance.")

st.info("""
**For real-time performance, you need GPU deployment:**
- Hugging Face Spaces (free GPU)
- Google Colab Pro  
- AWS/GCP with GPU instances
- Local machine with NVIDIA GPU
""")