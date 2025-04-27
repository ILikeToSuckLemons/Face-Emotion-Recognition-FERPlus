import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, RTCAppState
from streamlit_webrtc import RTCWE

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
    
if 'was_playing' not in st.session_state:
    st.session_state.was_playing = False
    
if 'face_detected' not in st.session_state:
    st.session_state.face_detected = False
    
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""

# Load model only once
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.info(f"Using device: {device}")
        model_path = "model/ferplus_model_pd_acc.pth"
        model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Global variables
model, device = load_model()
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Load face detection cascade with error checking
try:
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    st.sidebar.success("Face detection model loaded successfully")
except Exception as e:
    st.error(f"Error loading face detection: {str(e)}")
    haar_cascade = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

try:
    gif_overlay = GifEmotionOverlay("EmojiGif/")
    st.sidebar.success("GIF overlay loaded successfully")
except Exception as e:
    st.error(f"Error loading GIF overlay: {str(e)}")
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

# Add performance options in sidebar
st.sidebar.title("Performance Settings")
detect_frequency = st.sidebar.slider("Face Detection Frequency", 1, 10, 3, 
                                    help="Higher values = less frequent detection = better performance")
resolution_factor = st.sidebar.slider("Resolution Scale", 0.5, 1.0, 0.75, 0.05,
                                     help="Lower values = smaller resolution = better performance")
show_all_emotions = st.sidebar.checkbox("Show All Emotions", True,
                                      help="Uncheck to only display top emotion for better performance")

# Add debug options
st.sidebar.title("Debug Options")
debug_mode = st.sidebar.checkbox("Show Debug Info", True)
face_detection_scale = st.sidebar.slider("Face Detection Scale Factor", 1.05, 1.5, 1.1, 0.05,
                                      help="Lower values detect more faces but slower")
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3, 
                                 help="Lower values detect more faces but may include false positives")

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
        self.debug_text = ""
        self.face_detected = False
        
    def recv(self, frame):
        try:
            # Start frame timing
            start_time = time.time()
            
            img = frame.to_ndarray(format="bgr24")
            
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
                # Add debug rectangle to show processing area
                if debug_mode:
                    cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 2)
                
                # Improve face detection with histogram equalization
                gray_eq = cv2.equalizeHist(gray)
                
                self.faces = haar_cascade.detectMultiScale(
                    gray_eq, 
                    scaleFactor=face_detection_scale,
                    minNeighbors=min_neighbors, 
                    minSize=(int(30 * resolution_factor), int(30 * resolution_factor))
                )
                
                if len(self.faces) > 0:
                    self.face_detected = True
                    st.session_state.face_detected = True
                
            self.frame_count += 1
            
            # Scale back face coordinates to original image size
            scaled_faces = [(int(x * scale_factor), int(y * scale_factor), 
                             int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in self.faces]
            
            # Store emotion data for this frame
            current_time = time.time()
            frame_emotions = {emotion: 0 for emotion in emotions}
            
            # Show debug info about faces detected
            if debug_mode:
                self.debug_text = f"Detected {len(scaled_faces)} faces"
                cv2.putText(img, self.debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            
            for (x, y, w, h) in scaled_faces:
                try:
                    # Extract face from original image
                    face_region = img[y:y+h, x:x+w]
                    if face_region.size == 0:  # Skip if face region is invalid
                        continue
                        
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    
                    # Show the processed face in debug mode
                    if debug_mode:
                        # Place the small face in the corner for debugging
                        face_display = cv2.resize(face_resized, (96, 96))
                        img[10:106, w-106:w-10] = cv2.cvtColor(face_display, cv2.COLOR_GRAY2BGR)
                    
                    face_tensor = transform(face_resized).unsqueeze(0).to(device)
                    
                    # Run model inference
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                        top_emotion_idx = np.argmax(probs)
                        top_emotion = emotions[top_emotion_idx]
                    
                    # Record emotion data for all faces
                    for i, prob in enumerate(probs):
                        frame_emotions[emotions[i]] += prob
                    
                    # Animation offset logic
                    self.animation_offset += self.offset_direction * 2
                    if abs(self.animation_offset) > 10:
                        self.offset_direction *= -1
                    
                    # Overlay character if gif_overlay is available
                    if gif_overlay:
                        img = gif_overlay.overlay_gif(img, top_emotion, x, y, w, h, self.animation_offset)
                    
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
                    if debug_mode:
                        error_msg = f"Error in face processing: {str(e)}"
                        cv2.putText(img, error_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    continue
                    
            # Store emotion data if faces were detected
            if len(scaled_faces) > 0:
                # Calculate average emotions across all detected faces
                avg_emotions = {k: v/len(scaled_faces) if len(scaled_faces) > 0 else 0 for k, v in frame_emotions.items()}
                
                # Add to emotion history with relative time
                self.emotion_history.append({
                    'timestamp': current_time - self.start_time,  # Store as relative time
                    **avg_emotions
                })
                
                # Update session state for later use
                st.session_state.emotion_data = self.emotion_history
            
            # Calculate and display FPS if in debug mode
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            
            if debug_mode:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(img, fps_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Update debug info in session state
                st.session_state.debug_info = (
                    f"FPS: {fps:.1f}, Faces: {len(scaled_faces)}, "
                    f"Resolution: {w}x{h}, Scale: {resolution_factor:.2f}"
                )
        
        except Exception as e:
            # Global error handling
            error_msg = f"Error in frame processing: {str(e)}"
            st.session_state.debug_info = error_msg
            if img is not None:
                cv2.putText(img, "ERROR", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to create and display graphs
def display_emotion_graphs():
    if len(st.session_state.emotion_data) == 0:
        st.warning("No emotion data recorded. Start the webcam and show your face to record emotions.")
        return False
    
    # Convert emotion data to dataframe
    df = pd.DataFrame(st.session_state.emotion_data)
    
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
    
    return True

# Configure WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Display status info
status_placeholder = st.empty()

# Display WebRTC component


# Display WebRTC component
webrtc_ctx = webrtc_streamer(
    key="facial-emotion",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"max": 30}}, "audio": False},
    async_processing=True,
)

# Auto-display graphs when camera stops
if webrtc_ctx.state.playing:
    # Camera is running - set flag
    st.session_state.was_playing = True
elif st.session_state.get('was_playing', False):
    # Camera was playing but has now stopped - show graphs automatically
    st.session_state.was_playing = False
    st.session_state.show_graphs = True
    # Force a rerun to show the graphs immediately
    st.experimental_rerun()

# Show graphs when requested and we have data
if st.session_state.show_graphs:
    graphs_displayed = display_emotion_graphs()
    
    # Add a button to clear data if graphs were displayed
    if graphs_displayed and st.button("Clear Emotion Data"):
        st.session_state.emotion_data = []
        st.session_state.show_graphs = False
        st.experimental_rerun()


# Display data count
if st.session_state.emotion_data:
    st.sidebar.success(f"Data points collected: {len(st.session_state.emotion_data)}")

with st.expander("About this app"):
    st.write("""
    This app performs real-time facial emotion recognition using a trained deep learning model.
    It detects faces and classifies emotions into 8 categories: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, and Contempt.
    
    The app overlays emotion-specific GIFs and displays the probability for each emotion.
    
    The app will automatically generate emotion graphs when you stop the camera. You can also:
    - Press the "Generate Emotion Graphs" button anytime to see the current results
    - Clear collected data with the "Clear Emotion Data" button
    
    For better performance, adjust the settings in the sidebar:
    - Lower the face detection frequency
    - Reduce the resolution scale
    - Disable showing all emotions
    """)

with st.expander("Troubleshooting"):
    st.write("""
    If face detection is not working:
    
    1. Make sure your face is well-lit and clearly visible
    2. Try adjusting the Face Detection Scale Factor (lower values can detect more faces)
    3. Try lowering the Min Neighbors value (3-5 works well in most cases)
    4. Check that your camera is working properly
    5. Try enabling Debug Mode to see what's happening
    
    If the emotion recognition seems inaccurate:
    
    1. Ensure your face is well-lit from the front
    2. Try to keep a neutral pose and then express different emotions clearly
    3. The model works best with clear, exaggerated expressions
    """)