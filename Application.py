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
import threading


# Set page config
st.set_page_config(page_title="Real-time Facial Emotion Recognition", layout="wide")
st.title("Real-time Facial Emotion Recognition")

# Initialize session state for emotion data storage
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []
    
if 'show_graphs' not in st.session_state:
    st.session_state.show_graphs = True

# Add performance options in sidebar
st.sidebar.title("Performance Settings")
detect_frequency = st.sidebar.slider("Face Detection Frequency", 1, 10, 5, 
                                    help="Higher values = less frequent detection = better performance")
resolution_factor = st.sidebar.slider("Resolution Scale", 0.5, 1.0, 0.75, 0.05,
                                     help="Lower values = smaller resolution = better performance")
show_all_emotions = st.sidebar.checkbox("Show All Emotions", True,
                                      help="Uncheck to only display top emotion for better performance")

# Add graph settings
st.sidebar.title("Graph Settings")
show_graphs = st.sidebar.checkbox("Show Emotion Graphs", True)
st.session_state.show_graphs = show_graphs

# Load model only once
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "model/ferplus_model_pd_acc.pth"
        model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Global variables
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Try to load the model, with error handling
try:
    model, device = load_model()
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    gif_overlay = GifEmotionOverlay("EmojiGif/")
except Exception as e:
    st.error(f"Error initializing components: {str(e)}")
    model, device = None, None
    haar_cascade = None
    transform = None
    gif_overlay = None

# Emotion color definitions
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

# Lockable emotion data
emotion_data_lock = threading.Lock()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.color_index = 0
        self.animation_offset = 0
        self.offset_direction = 1
        self.faces = []
        self.last_frame_time = 0
        self.processing_fps = 0
        self.emotion_history = []
        self.start_time = time.time()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            if haar_cascade is None or model is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Downscale image for processing (improves performance)
            h, w = img.shape[:2]
            img_small = cv2.resize(img, (int(w * resolution_factor), int(h * resolution_factor)))
            scale_factor = 1 / resolution_factor
            
            # Process frame
            gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            
            # Color cycling
            if self.frame_count % 10 == 0:  
                self.color_index = (self.color_index + 1) % 3
                
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
            
            # Process faces and emotions
            current_time = time.time()
            frame_emotions = {emotion: 0 for emotion in emotions}
            faces_processed = 0
            
            for (x, y, w, h) in scaled_faces:
                # Ensure face coordinates are within image bounds
                if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                    continue
                    
                # Extract face from original image
                face_region = img[y:y+h, x:x+w]
                if face_region.size == 0:  # Skip if face region is invalid
                    continue
                    
                try:
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    face_tensor = torch.tensor(face_resized, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Run model inference
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                        top_emotion_idx = np.argmax(probs)
                        top_emotion = emotions[top_emotion_idx]
                    
                    # Record emotion data for this face
                    for i, prob in enumerate(probs):
                        frame_emotions[emotions[i]] += prob
                    
                    faces_processed += 1
                    
                    # Animation offset logic for GIF overlay
                    self.animation_offset += self.offset_direction * 2
                    if abs(self.animation_offset) > 10:
                        self.offset_direction *= -1
                    
                    # Overlay character
                    try:
                        img = gif_overlay.overlay_gif(img, top_emotion, x, y, w, h, self.animation_offset)
                    except Exception as e:
                        # Fallback to simple rectangle if overlay fails
                        pass
                    
                    # Draw rectangle
                    box_color = emotion_colors.get(top_emotion, (255, 255, 255))
                    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
                    
                    # Display either all emotions or just the top one based on settings
                    if show_all_emotions:
                        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                            if i == top_emotion_idx:
                                text_color = emotion_text_colors[top_emotion][self.color_index]
                            else:
                                text_color = (255, 255, 255)
                            
                            text = f"{emotion}: {int(prob * 100)}%"
                            cv2.putText(img, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    else:
                        # Only show top emotion
                        text_color = emotion_text_colors[top_emotion][self.color_index]
                        text = f"{top_emotion}: {int(probs[top_emotion_idx] * 100)}%"
                        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                        
                except Exception as e:
                    # Skip this face if processing fails
                    continue
            
            # Update emotion data if faces were processed
            if faces_processed > 0:
                # Calculate average emotions across processed faces
                avg_emotions = {k: v/faces_processed for k, v in frame_emotions.items()}
                
                # Create data point with timestamp
                emotion_data_point = {
                    'timestamp': current_time - self.start_time,
                    **avg_emotions
                }
                
                # Safely update emotion history
                self.emotion_history.append(emotion_data_point)
                
                # Only keep last 100 data points to avoid memory issues
                if len(self.emotion_history) > 100:
                    self.emotion_history = self.emotion_history[-100:]
                
                # Update session state safely
                with emotion_data_lock:
                    st.session_state.emotion_data = self.emotion_history.copy()
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Return original frame if any error occurs
            if frame is not None:
                return frame
            # Create blank frame if all else fails
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(blank, format="bgr24")

# Function to create and display graphs
def display_emotion_graphs():
    with emotion_data_lock:
        emotion_data = st.session_state.emotion_data.copy() if 'emotion_data' in st.session_state else []
    
    if len(emotion_data) == 0:
        st.warning("No emotion data recorded yet. The graphs will appear once facial emotions are detected.")
        return False
    
    try:
        # Convert emotion data to dataframe
        df = pd.DataFrame(emotion_data)
        
        # Create two columns for the graphs
        col1, col2 = st.columns(2)
        
        with col1:
            # Create bar chart of total emotion scores
            st.subheader("Total Emotion Distribution")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            
            # Calculate average emotions across all frames
            avg_emotions = df[emotions].mean().sort_values(ascending=False)
            
            # Create bar chart with custom colors
            bars = ax1.bar(avg_emotions.index, avg_emotions.values * 100)
            
            # Set colors for bars
            for i, bar in enumerate(bars):
                emotion_name = avg_emotions.index[i]
                # Convert BGR to RGB
                bgr_color = emotion_colors[emotion_name]
                rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
                bar.set_color(rgb_color)
            
            ax1.set_ylabel('Average Score (%)')
            ax1.set_title('Average Emotion Distribution')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(avg_emotions.values):
                ax1.text(i, v * 100 + 1, f"{v*100:.1f}%", ha='center')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # Create line graph of emotions over time
            st.subheader("Emotions Over Time")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            
            # Plot each emotion line
            for emotion in emotions:
                # Convert BGR to RGB
                bgr_color = emotion_colors[emotion]
                rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
                
                ax2.plot(df['timestamp'], df[emotion] * 100, label=emotion, color=rgb_color, linewidth=2)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Emotion Score (%)')
            ax2.set_title('Emotion Scores Over Time')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Add a clear data button
        if st.button("Clear Emotion Data"):
            with emotion_data_lock:
                st.session_state.emotion_data = []
            st.experimental_rerun()
        
        return True
    except Exception as e:
        st.error(f"Error displaying graphs: {str(e)}")
        return False

# Configure WebRTC with minimal settings
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Display WebRTC component
webrtc_ctx = webrtc_streamer(
    key="facial-emotion",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Create graphs placeholder
graph_placeholder = st.empty()

# Display graphs if enabled
if st.session_state.show_graphs:
    with graph_placeholder:
        display_emotion_graphs()

# Refresh graphs every 3 seconds if WebRTC is active
if webrtc_ctx.state.playing and st.session_state.show_graphs:
    refresh_rate = 3
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time() - refresh_rate
    
    if time.time() - st.session_state.last_refresh > refresh_rate:
        st.session_state.last_refresh = time.time()
        with graph_placeholder:
            display_emotion_graphs()

with st.expander("About this app"):
    st.write("""
    This app performs real-time facial emotion recognition using a trained deep learning model.
    It detects faces and classifies emotions into 8 categories: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, and Contempt.
    
    The app overlays emotion-specific GIFs and displays the probability for each emotion.
    
    The graphs will automatically show:
    1. A bar chart of the total emotion distribution
    2. A line graph showing how emotions changed over time
    
    For better performance, adjust the settings in the sidebar:
    - Lower the face detection frequency
    - Reduce the resolution scale
    - Disable showing all emotions
    - Turn off graphs temporarily if needed
    """)