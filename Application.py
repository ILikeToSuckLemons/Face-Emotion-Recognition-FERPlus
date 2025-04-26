import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from models import PerformanceModel

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

# Create simple sidebar controls
st.sidebar.title("Performance Settings")
process_every_n = st.sidebar.slider("Process every N frames", 1, 10, 3)
face_confidence = st.sidebar.slider("Face detection confidence", 1, 10, 5)
resolution_scale = st.sidebar.slider("Processing resolution", 30, 100, 50)

# Load model only once
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
    model.eval()
    return model, device

# Load face detector only once
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load resources
model, device = load_model()
face_detector = load_face_detector()

# Define the video processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_faces = []
        self.last_emotions = []
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Only process every N frames to reduce CPU usage
        if self.frame_count % process_every_n == 0:
            # Reduce resolution for faster processing
            h, w = img.shape[:2]
            scale = resolution_scale / 100.0
            small_frame = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with adjusted parameters for speed
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Larger scale factor = faster but less accurate
                minNeighbors=face_confidence,  # Adjust based on sidebar control
                minSize=(20, 20)  # Minimum face size to detect
            )
            
            # Scale face coordinates back to original size
            self.last_faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) 
                              for (x, y, w, h) in faces]
            
            # Clear previous emotions
            self.last_emotions = []
            
            # Only process emotions if faces are detected
            if len(self.last_faces) > 0:
                for (x, y, w, h) in self.last_faces:
                    # Ensure face coordinates are within image bounds
                    x, y = max(0, x), max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    if w <= 0 or h <= 0:
                        self.last_emotions.append("Unknown")
                        continue
                    
                    # Extract face ROI
                    face_roi = img[y:y+h, x:x+w]
                    
                    # Convert to grayscale and resize to model input size
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    
                    # Prepare tensor
                    face_tensor = torch.tensor(face_resized, dtype=torch.float32) / 255.0
                    face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
                    face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Get emotion prediction (in a safer way)
                    try:
                        with torch.no_grad():
                            outputs = model(face_tensor)
                            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                            emotion = emotions[np.argmax(probs)]
                            self.last_emotions.append(emotion)
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")
                        self.last_emotions.append("Error")
        
        # Draw the most recent detection results
        for i, (x, y, w, h) in enumerate(self.last_faces):
            if i < len(self.last_emotions):
                emotion = self.last_emotions[i]
                color = emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add a frame counter in the corner
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-Time Facial Emotion Recognition")
st.write("""
This app detects emotions in real-time using your webcam.
Adjust the settings in the sidebar to reduce lag if needed.
""")

# Create WebRTC streamer with minimal options
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    # Remove async_processing to avoid asyncio errors
)