import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page config with a focused layout
st.set_page_config(
    page_title="Real-Time Emotion Detection",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for more video space
)

# Define the PerformanceModel class (simplified for faster loading)
class PerformanceModel(torch.nn.Module):
    def __init__(self, input_shape=(1, 48, 48), n_classes=8):
        super(PerformanceModel, self).__init__()
        # Use smaller layers for faster inference
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flat_features = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)
        
        # Simplified fully connected layers
        self.dropout = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(self.flat_features, 512)
        self.fc2 = torch.nn.Linear(512, n_classes)
    
    def forward(self, x):
        # Forward pass with minimal operations
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.flat_features)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Create global variables for performance
if 'model' not in st.session_state:
    st.session_state.model = None
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cpu")  # Use CPU for compatibility
if 'transform' not in st.session_state:
    st.session_state.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
if 'emotions' not in st.session_state:
    st.session_state.emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
if 'detection_interval' not in st.session_state:
    st.session_state.detection_interval = 2  # Process every N frames
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.4  # Lower threshold for better detection

# Define emotion colors
emotion_colors = {
    "Neutral": (255, 255, 255),    # White
    "Happy": (0, 255, 255),        # Yellow
    "Surprise": (0, 165, 255),     # Orange
    "Sad": (255, 0, 0),            # Blue
    "Angry": (0, 0, 255),          # Red
    "Disgust": (128, 0, 128),      # Purple
    "Fear": (255, 255, 0),         # Cyan
    "Contempt": (0, 255, 0)        # Green
}

# WebRTC Video Processor for real-time processing
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.faces = []
        self.model = st.session_state.model
        self.face_cascade = st.session_state.face_cascade
        self.transform = st.session_state.transform
        self.device = st.session_state.device
        self.emotions = st.session_state.emotions
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Convert to grayscale (faster processing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Only run face detection every N frames for speed
        if self.frame_count % st.session_state.detection_interval == 0:
            self.faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,  # Larger scale factor = faster but less accurate
                minNeighbors=4,   # Lower value = more detections but possible false positives
                minSize=(30, 30)  # Smaller minimum size
            )
            
        # Process each detected face
        for (x, y, w, h) in self.faces:
            # Extract and preprocess face
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            
            # Convert to tensor and run inference
            face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():  # Disable gradient calculation for speed
                outputs = self.model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_emotion_idx = np.argmax(probs)
                top_emotion = self.emotions[top_emotion_idx]
                top_prob = probs[top_emotion_idx]
            
            # Only display if confidence exceeds threshold
            if top_prob >= st.session_state.confidence_threshold:
                # Draw rectangle with emotion color
                color = emotion_colors.get(top_emotion, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Display emotion text (minimal text for speed)
                text = f"{top_emotion}"
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
        
        # Add frame counter (useful for debugging performance)
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img)

def main():
    st.title("Real-Time Facial Emotion Recognition")
    
    # Simple UI with focus on performance
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Settings")
        
        # Performance settings
        st.session_state.detection_interval = st.slider(
            "Detection Interval", 1, 10, st.session_state.detection_interval,
            help="Higher values = better performance, lower accuracy"
        )
        
        st.session_state.confidence_threshold = st.slider(
            "Confidence", 0.0, 1.0, st.session_state.confidence_threshold,
            help="Higher values = fewer detections, more certainty"
        )
        
        # Model uploader
        uploaded_model = st.file_uploader("Upload model file (ferplus_model_pd_acc.pth)", type=["pth"])
        
        # Load model when uploaded
        model_status = st.empty()
        
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
                
            try:
                # Load the model with minimal overhead
                st.session_state.model = PerformanceModel(
                    input_shape=(1, 48, 48), 
                    n_classes=8
                ).to(st.session_state.device)
                
                st.session_state.model.load_state_dict(
                    torch.load(model_path, map_location=st.session_state.device),
                    strict=False  # Allow partial loading for robustness
                )
                st.session_state.model.eval()  # Set to evaluation mode
                
                # Use torch's JIT compilation for speed if available
                try:
                    st.session_state.model = torch.jit.script(st.session_state.model)
                    model_status.success("Model loaded with JIT optimization!")
                except:
                    model_status.success("Model loaded successfully!")
            except Exception as e:
                model_status.error(f"Error loading model: {e}")
                st.session_state.model = None
    
    with col1:
        # Simple explanation
        st.info("This app performs real-time facial emotion detection. Click 'Start' below to begin.")
        
        # Check if model is loaded before showing webcam
        if st.session_state.model is None:
            st.warning("Please upload a model file first")
            return
            
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Start WebRTC streamer with minimal configuration
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,  # Enable async for better performance
        )
        
        # Show helpful information when streaming
        if webrtc_ctx.state.playing:
            st.markdown("""
            **Tips for best performance:**
            - Ensure good lighting
            - Look directly at the camera
            - If performance is slow, increase the detection interval
            """)

# Clean up temp files
def cleanup():
    for file in os.listdir(tempfile.gettempdir()):
        if file.endswith('.pth'):
            try:
                os.unlink(os.path.join(tempfile.gettempdir(), file))
            except:
                pass

# Run the app
if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()