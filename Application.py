import cv2
import torch
import numpy as np
import torch.nn.functional as F
import streamlit as st
from torchvision import transforms
import os
import time
from threading import Thread

# Set page configuration
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

# Define the PerformanceModel class (since we need it in Streamlit)
class PerformanceModel(torch.nn.Module):
    def __init__(self, input_shape, n_classes, logits=True):
        super(PerformanceModel, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.logits = logits
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(128 * 6 * 6, 512)
        self.fc2 = torch.nn.Linear(512, n_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        if not self.logits:
            x = F.softmax(x, dim=1)
        
        return x

# Define the EmotionOverlay (placeholder for the imported class)
class GifEmotionOverlay:
    def __init__(self, gif_folder):
        self.emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
        self.gif_folder = gif_folder
        self.frames = {}
        
        # Create a dictionary to store the frames for each emotion (placeholder)
        for emotion in self.emotions:
            self.frames[emotion] = [np.zeros((100, 100, 3), dtype=np.uint8)]  # Placeholder

    def overlay_gif(self, frame, emotion, x, y, w, h, animation_offset):
        # Simplified overlay function for the Streamlit version
        # Draw a colored rectangle based on emotion instead
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
        
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.putText(frame, emotion, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

# App initialization
@st.cache_resource
def load_model():
    # Load model with caching to avoid reloading on each rerun
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    
    # Check if the model file exists or use a placeholder model for testing
    model_path = "model/ferplus_model_pd_acc.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("Model file not found. Using untrained model for demo.")
    
    model.eval()
    return model, device

def process_frame(frame, model, device, gif_overlay):
    # Process the frame for emotion detection
    emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
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
    
    # Convert to RGB (for Streamlit) and grayscale (for processing)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    
    color_index = int(time.time() * 0.3) % 3  # Cycle through colors based on time
    animation_offset = int(10 * np.sin(time.time() * 2))  # Smooth animation
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        
        # Prepare for model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        face_tensor = transform(face_img).unsqueeze(0).to(device)
        
        # Run model inference
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]
        
        # Overlay emotion
        frame_rgb = gif_overlay.overlay_gif(frame_rgb, top_emotion, x, y, w, h, animation_offset)
        
        # Draw rectangle around face
        box_color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), box_color, 2)
        
        # Display emotions with probabilities
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            if i == top_emotion_idx:
                text_color = emotion_text_colors[top_emotion][color_index]
            else:
                text_color = (255, 255, 255)
            
            text = f"{emotion}: {int(prob * 100)}%"
            cv2.putText(frame_rgb, text, (x, y - 10 - (i * 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return frame_rgb

def main():
    st.title("Facial Emotion Recognition")
    
    # Load the model
    model, device = load_model()
    
    # Initialize GIF overlay
    gif_folder = "EmojiGif/"
    gif_overlay = GifEmotionOverlay(gif_folder)
    
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    # Session state to track if webcam is running
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False
    
    # Start/Stop webcam button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button('Start' if not st.session_state.run_webcam else 'Stop'):
            st.session_state.run_webcam = not st.session_state.run_webcam
    
    # Create and use a Streamlit webrtc component for real-time video
    if st.session_state.run_webcam:
        try:
            # Use opencv-python-headless for Streamlit Cloud compatibility
            cap = cv2.VideoCapture(0)
            
            # Process frames in a loop
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                # Process the frame
                processed_frame = process_frame(frame, model, device, gif_overlay)
                
                # Display the processed frame
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Control frame rate (adjust for performance)
                time.sleep(0.03)  # ~30 FPS, adjust for smoother performance
            
            # Release resources when stopped
            cap.release()
            
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
            st.info("Streamlit Cloud may have limited webcam support. Try running locally.")
    else:
        # Display a message when webcam is not active
        frame_placeholder.info("Click 'Start' to activate the webcam for emotion recognition")

    # Add some usage instructions
    with st.expander("Instructions"):
        st.write("""
        1. Click 'Start' to begin emotion recognition from your webcam
        2. The app will detect faces and display the detected emotion
        3. Click 'Stop' to end the webcam feed
        
        Note: For best performance, ensure good lighting and face the camera directly.
        """)

if __name__ == "__main__":
    main()