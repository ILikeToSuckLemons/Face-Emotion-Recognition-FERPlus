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

# Define the video processor
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_roi, (48, 48))
            face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5)
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_idx = np.argmax(probs)
                emotion = emotions[top_idx]
            
            color = emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-Time Facial Emotion Recognition")

webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
