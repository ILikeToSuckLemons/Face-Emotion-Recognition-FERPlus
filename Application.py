import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import tempfile
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page config first
st.set_page_config(page_title="Real-Time Emotion Recognition", layout="wide")

# Define the PerformanceModel class (since you need this from your original models.py)
class PerformanceModel(torch.nn.Module):
    def __init__(self, input_shape=(1, 48, 48), n_classes=8, logits=True):
        super(PerformanceModel, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.logits = logits
        
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        
        # Calculate the flattened features size
        self.flat_features = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)
        
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(self.flat_features, 1024)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(-1, self.flat_features)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'emotions' not in st.session_state:
    st.session_state.emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if 'transform' not in st.session_state:
    st.session_state.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
if 'detection_frequency' not in st.session_state:
    st.session_state.detection_frequency = 3
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

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

# WebRTC video processor for real-time processing
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.faces = []
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Update frame count
        self.frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Run face detection based on frequency
        if self.frame_count % st.session_state.detection_frequency == 0:
            self.faces = st.session_state.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5, 
                minSize=(48, 48)
            )
        
        # Process each detected face
        for (x, y, w, h) in self.faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            
            # Convert to tensor
            face_tensor = st.session_state.transform(face_img).unsqueeze(0).to(st.session_state.device)
            
            # Run model inference
            if st.session_state.model:
                with torch.no_grad():
                    outputs = st.session_state.model(face_tensor)
                    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                    top_emotion_idx = np.argmax(probs)
                    top_emotion = st.session_state.emotions[top_emotion_idx]
                    top_prob = probs[top_emotion_idx]
                
                # Only display if confidence exceeds threshold
                if top_prob >= st.session_state.confidence_threshold:
                    # Draw rectangle with emotion color
                    color = emotion_colors.get(top_emotion, (255, 255, 255))
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    
                    # Display emotion text
                    text = f"{top_emotion}: {int(top_prob * 100)}%"
                    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img)

def main():
    st.title("Real-Time Facial Emotion Recognition")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    
    # Update detection frequency
    st.session_state.detection_frequency = st.sidebar.slider(
        "Detection Frequency (frames)", 
        1, 10, st.session_state.detection_frequency,
        help="How often to run face detection. Higher values improve performance."
    )
    
    # Update confidence threshold
    st.session_state.confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, st.session_state.confidence_threshold,
        help="Minimum confidence to display an emotion"
    )
    
    # App explanation
    with st.expander("About this app"):
        st.write("""
        This app performs real-time facial emotion recognition using WebRTC for low-latency video processing.
        It can recognize 8 emotions: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, and Contempt.
        
        For the best performance:
        - Use Google Chrome or Firefox
        - Adjust the detection frequency based on your device's performance
        - Make sure your face is well-lit
        """)
    
    # Model upload section
    uploaded_model = st.sidebar.file_uploader("Upload model file (ferplus_model_pd_acc.pth)", type=["pth"])
    model_status = st.empty()
    
    # Load model if uploaded
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path = tmp_file.name
            
        try:
            st.session_state.model = PerformanceModel(
                input_shape=(1, 48, 48), 
                n_classes=8, 
                logits=True
            ).to(st.session_state.device)
            
            st.session_state.model.load_state_dict(
                torch.load(model_path, map_location=st.session_state.device)
            )
            st.session_state.model.eval()
            model_status.success(f"Model loaded successfully! Using {st.session_state.device}")
        except Exception as e:
            model_status.error(f"Error loading model: {e}")
            st.session_state.model = None
    else:
        model_status.warning("Please upload your model file to continue")
    
    # App modes
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Real-Time WebRTC", "Upload Image"])
    
    if app_mode == "Real-Time WebRTC":
        st.subheader("Live Webcam Feed (WebRTC)")
        
        # Check if model is loaded
        if st.session_state.model is None:
            st.warning("Please upload a model before starting the webcam")
            return
        
        # Configure WebRTC
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Start WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Instructions
        if webrtc_ctx.video_processor:
            st.info("Webcam is active! Make sure you allow camera access.")
        
        # Add statistics display
        if webrtc_ctx.state.playing:
            st.subheader("Detection Stats")
            stats_cols = st.columns(2)
            with stats_cols[0]:
                st.metric("Device", f"{st.session_state.device}")
            with stats_cols[1]:
                st.metric("Detection Frequency", f"Every {st.session_state.detection_frequency} frames")
    
    elif app_mode == "Upload Image":
        st.subheader("Image Upload")
        
        # Check if model is loaded
        if st.session_state.model is None:
            st.warning("Please upload a model before processing images")
            return
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert to BGR for OpenCV processing
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                # Handle grayscale images
                image_bgr = image_np
                if len(image_bgr.shape) == 2:
                    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Emotions"):
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = st.session_state.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(48, 48)
                )
                
                # Create a copy of the image for results
                result_image = image_bgr.copy()
                
                if len(faces) == 0:
                    st.warning("No faces detected in the image")
                else:
                    st.write(f"Found {len(faces)} face(s)")
                    
                    all_emotions = []
                    
                    for (x, y, w, h) in faces:
                        # Extract and preprocess face
                        face_img = gray[y:y + h, x:x + w]
                        face_img = cv2.resize(face_img, (48, 48))
                        
                        # Convert to tensor
                        face_tensor = st.session_state.transform(face_img).unsqueeze(0).to(st.session_state.device)
                        
                        # Run model inference
                        with torch.no_grad():
                            outputs = st.session_state.model(face_tensor)
                            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                        
                        # Get top emotion
                        top_emotion_idx = np.argmax(probs)
                        top_emotion = st.session_state.emotions[top_emotion_idx]
                        top_prob = probs[top_emotion_idx]
                        
                        # Store results
                        all_emotions.append({
                            "face_id": len(all_emotions) + 1,
                            "emotion": top_emotion,
                            "confidence": top_prob,
                            "all_probs": probs
                        })
                        
                        # Draw rectangle with emotion color
                        color = emotion_colors.get(top_emotion, (255, 255, 255))
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                        
                        # Add text with top emotion and face ID
                        text = f"Face #{len(all_emotions)}: {top_emotion} ({int(top_prob * 100)}%)"
                        cv2.putText(result_image, text, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert back to RGB for display
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Display result image
                    st.image(result_rgb, caption="Detection Result", use_column_width=True)
                    
                    # Show detailed results for each face
                    for face_data in all_emotions:
                        with st.expander(f"Face #{face_data['face_id']} - {face_data['emotion']}"):
                            # Create columns for metrics
                            cols = st.columns(2)
                            cols[0].metric("Top Emotion", face_data['emotion'])
                            cols[1].metric("Confidence", f"{face_data['confidence']:.2%}")
                            
                            # Show all emotions as bar chart
                            emotion_data = {
                                "Emotion": st.session_state.emotions,
                                "Probability": face_data['all_probs']
                            }
                            st.bar_chart(emotion_data, x="Emotion", y="Probability")

# Clean up temporary files on app shutdown
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