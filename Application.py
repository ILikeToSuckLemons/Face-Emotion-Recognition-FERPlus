import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import time
import matplotlib.pyplot as plt

# Import your custom modules
from models import PerformanceModel
from gifoverlay import GifEmotionOverlay

# App title and description
st.title("Facial Emotion Recognition")
st.write("Upload an image or use your webcam to detect emotions in real-time!")

# Sidebar for app controls
st.sidebar.header("Settings")
detection_mode = st.sidebar.radio("Choose Detection Mode", ["Upload Image", "Webcam"])

# Performance settings
if 'enable_gif' not in st.session_state:
    st.session_state.enable_gif = True
if 'process_every_n_frames' not in st.session_state:
    st.session_state.process_every_n_frames = 2
if 'frame_scale' not in st.session_state:
    st.session_state.frame_scale = 0.75

# Performance settings in sidebar
st.sidebar.subheader("Performance Settings")
st.session_state.enable_gif = st.sidebar.checkbox("Enable GIF Overlay", value=st.session_state.enable_gif)
st.session_state.process_every_n_frames = st.sidebar.slider("Process every N frames", 1, 5, st.session_state.process_every_n_frames)
st.session_state.frame_scale = st.sidebar.slider("Frame Scale", 0.5, 1.0, st.session_state.frame_scale, 0.05)

# Define emotions list
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Initialize session state variables
if "emotion_over_time" not in st.session_state:
    st.session_state.emotion_over_time = {emotion: [] for emotion in emotions}
    st.session_state.frame_timestamps = []
    st.session_state.run_webcam = False
    st.session_state.animation_offset = 0
    st.session_state.offset_direction = 1
    st.session_state.should_plot = False
    st.session_state.color_index = 0
    st.session_state.frame_count = 0
    st.session_state.last_prediction = None

# Define emotion colors for OpenCV (BGR in 0-255 range)
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

# Define emotion colors for matplotlib (RGB in 0-1 range)
emotion_colors_mpl = {
    "Neutral": (0.7, 0.7, 0.7),     # White
    "Happy": (1.0, 1.0, 0.0),       # Yellow
    "Surprise": (1.0, 0.65, 0.0),   # Orange
    "Sad": (0.0, 0.0, 1.0),         # Blue
    "Angry": (1.0, 0.0, 0.0),       # Red
    "Disgust": (0.5, 0.0, 0.5),     # Purple
    "Fear": (0.0, 1.0, 1.0),        # Cyan
    "Contempt": (0.0, 1.0, 0.0)     # Green
}

# Define emotion text colors (simplified to single colors for performance)
emotion_text_colors = {
    "Neutral": (128, 128, 128),
    "Happy": (76, 235, 253),
    "Surprise": (247, 255, 0),
    "Sad": (194, 105, 3),
    "Angry": (61, 57, 242),
    "Disgust": (70, 190, 77),
    "Fear": (198, 128, 134),
    "Contempt": (160, 134, 72)
}

# Load model function - @st.cache_resource prevents reloading on every rerun
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    try:
        model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

# Load GIF emotion overlay
@st.cache_resource
def load_gif_overlay():
    try:
        return GifEmotionOverlay("EmojiGif/")
    except Exception as e:
        st.warning(f"Could not load GIF overlays: {str(e)}")
        return None

# Get face detector
@st.cache_resource
def load_face_detector():
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None

# Emotion prediction function
def predict_emotion(face_tensor, model):
    try:
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]
        return probs, top_emotion, top_emotion_idx
    except Exception as e:
        st.error(f"Error during emotion prediction: {str(e)}")
        # Return fallback values
        fallback_probs = np.zeros(len(emotions))
        fallback_probs[0] = 1.0  # Default to Neutral
        return fallback_probs, "Neutral", 0

# Process image function - optimized version
def process_image(img, model, device, face_detector, gif_overlay=None):
    try:
        # Scale down image for faster processing
        scale = st.session_state.frame_scale
        if scale != 1.0:
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        else:
            img_small = img.copy()
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.3,  # Slightly increased for speed
            minNeighbors=3,   # Reduced for better detection rate
            minSize=(int(48 * scale), int(48 * scale))
        )
        
        detected_faces = []
        # Scale back face coordinates to original image size
        if scale != 1.0:
            faces_original = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]
        else:
            faces_original = faces
        
        # Store last prediction for smoother display
        probs = None
        top_emotion = None
        top_emotion_idx = None
        
        # Process each face
        for (x, y, w, h) in faces_original:
            # Extract and preprocess face from the small image
            face_roi = gray[int(y*scale):int((y+h)*scale), int(x*scale):int((x+w)*scale)]
            if face_roi.size == 0:
                continue
                
            # Resize to model input size
            face_img = cv2.resize(face_roi, (48, 48))
            face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
            
            # Get prediction
            probs, top_emotion, top_emotion_idx = predict_emotion(face_tensor, model)
            st.session_state.last_prediction = (probs, top_emotion, top_emotion_idx)
            
            # Store detected face info
            detected_faces.append((x, y, w, h, top_emotion, probs))
            
            # Apply GIF overlay conditionally
            if gif_overlay and st.session_state.enable_gif:
                animation_offset = st.session_state.animation_offset
                try:
                    img_with_overlay = gif_overlay.overlay_gif(
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 
                        top_emotion, x, y, w, h, animation_offset
                    )
                    img_rgb = cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    pass  # Silently continue if overlay fails
            
            # Draw rectangle around face
            color = emotion_colors.get(top_emotion, (255, 255, 255))
            color_rgb = (color[2], color[1], color[0])  # Convert BGR to RGB
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color_rgb, 2)
            
            # Display ALL emotion probabilities - removed filtering condition
            for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                # Choose text color
                if i == top_emotion_idx:
                    text_color = emotion_text_colors[top_emotion]
                    text_color = (text_color[2], text_color[1], text_color[0])  # BGR to RGB
                else:
                    text_color = (255, 255, 255)  # White for other emotions

                text = f"{emotion}: {int(prob * 100)}%"
                
                # Position text to the LEFT of the face with padding
                text_width = len(text) * 10  # Approximate width of text
                text_x = max(10, x - text_width - 15)  # Left-align with padding (15px from face)
                text_y = y + 20 + (i * 20)  # Vertical stacking
                
                # Ensure text doesn't go off-screen (top/bottom)
                if 0 <= text_y < img_rgb.shape[0]:
                    cv2.putText(img_rgb, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return img_rgb, detected_faces
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        # Return a fallback image
        fallback_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(fallback_img, "Error processing image", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        fallback_img_rgb = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
        return fallback_img_rgb, []

# Update emotion data from detection results
def update_emotion_data(faces_info):
    if faces_info and len(faces_info) > 0:
        # Take the first face for tracking
        _, _, _, _, _, probs = faces_info[0]
        
        for i, emotion in enumerate(emotions):
            st.session_state.emotion_over_time[emotion].append(probs[i])
        
        st.session_state.frame_timestamps.append(len(st.session_state.frame_timestamps))

# Function to generate and display plots
def generate_plots():
    # Only generate plots if we have data
    if len(st.session_state.frame_timestamps) > 0:
        try:
            # Calculate the dominant emotion for each frame
            dominant_emotions = []
            for i in range(len(st.session_state.frame_timestamps)):
                frame_emotions = {emotion: st.session_state.emotion_over_time[emotion][i] 
                                if i < len(st.session_state.emotion_over_time[emotion]) else 0 
                                for emotion in emotions}
                dominant_emotions.append(max(frame_emotions, key=frame_emotions.get))
            
            # Plot 1: Total Emotion Scores
            emotion_totals = {
                emotion: sum(st.session_state.emotion_over_time[emotion])
                for emotion in emotions
            }

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            # Use matplotlib color format (0-1 range)
            bars = ax1.bar(emotion_totals.keys(), emotion_totals.values(), 
                           color=[emotion_colors_mpl[e] for e in emotions])
            ax1.set_title("Total Emotion Scores (Summed Over Time)")
            ax1.set_xlabel("Emotion")
            ax1.set_ylabel("Total Score")
            plt.xticks(rotation=45)
            ax1.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig1)

            # Plot 2: Emotion Scores Over Time
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            for emotion, scores in st.session_state.emotion_over_time.items():
                # Use matplotlib color format (0-1 range)
                rgb_color = emotion_colors_mpl[emotion]
                ax2.plot(st.session_state.frame_timestamps[:len(scores)], scores, 
                        label=emotion, linewidth=1.5, color=rgb_color)

            ax2.set_title("Emotion Scores Over Time")
            ax2.set_xlabel("Frame Number")
            ax2.set_ylabel("Probability")
            ax2.legend(loc="upper right")
            ax2.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Plot 3: Dominant Emotion Timeline
            fig3, ax3 = plt.subplots(figsize=(12, 4))
            unique_emotions = list(set(dominant_emotions))
            emotion_indices = {emotion: i for i, emotion in enumerate(unique_emotions)}
            
            # Create a scatter plot for dominant emotions
            for i, emotion in enumerate(dominant_emotions):
                # Use matplotlib color format (0-1 range)
                rgb_color = emotion_colors_mpl[emotion]
                ax3.scatter(st.session_state.frame_timestamps[i], emotion_indices[emotion], 
                           color=rgb_color, s=50, label=emotion if emotion not in ax3.get_legend_handles_labels()[1] else "")
            
            # Connect points with lines
            emotion_idx_values = [emotion_indices[emotion] for emotion in dominant_emotions]
            ax3.plot(st.session_state.frame_timestamps, emotion_idx_values, color='gray', alpha=0.3)
            
            ax3.set_yticks(range(len(unique_emotions)))
            ax3.set_yticklabels(unique_emotions)
            ax3.set_title("Dominant Emotion Timeline")
            ax3.set_xlabel("Frame Number")
            ax3.grid(True, linestyle='--', alpha=0.3)
            
            # Use a legend without duplicates
            handles, labels = [], []
            for h, l in zip(*ax3.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
            if handles:
                ax3.legend(handles, labels, loc="upper right")
                
            plt.tight_layout()
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error generating plots: {str(e)}")
    else:
        st.warning("No emotion data collected yet. Please run the webcam first.")

# Main app logic based on selection
webcam_available = True

if detection_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model and face detector
        model, device = load_model()
        face_detector = load_face_detector()
        gif_overlay = load_gif_overlay()
        
        if model is None or face_detector is None:
            st.error("Required components failed to load. Please check the error messages above.")
        else:
            try:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Process image
                result_img, faces = process_image(img, model, device, face_detector, gif_overlay)
                
                # Display result
                st.image(result_img, caption="Processed Image", use_container_width=True)
                
                if len(faces) == 0:
                    st.warning("No faces detected in the image!")
                else:
                    st.success(f"Detected {len(faces)} face(s) in the image!")
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")

elif detection_mode == "Webcam":
    try:
        # Try to import WebRTC components with proper error handling
        import av
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
        
        # Load model and face detector
        model, device = load_model()
        face_detector = load_face_detector()
        gif_overlay = load_gif_overlay()
        
        if model is None or face_detector is None:
            st.error("Required components failed to load. Please check the error messages above.")
            webcam_available = False
        else:
            # Setup WebRTC configuration with modifications for Streamlit Cloud
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=False  # Important for Streamlit Cloud
            )
            
            # Create a video transformer for emotion detection
            class EmotionDetector(VideoTransformerBase):
                def __init__(self):
                    self.frame_count = 0
                    self.animation_offset = 0
                    
                def transform(self, frame):
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Process every N frames for better performance
                        if self.frame_count % st.session_state.process_every_n_frames == 0:
                            # Update animation offset
                            self.animation_offset = (self.animation_offset + 1) % 10
                            st.session_state.animation_offset = self.animation_offset
                            
                            # Process frame
                            result_frame, faces = process_image(img, model, device, face_detector, gif_overlay)
                            
                            # Update emotion data if faces found
                            if faces:
                                update_emotion_data(faces)
                        else:
                            # For skipped frames, just convert to RGB directly
                            result_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Increment frame counter
                        self.frame_count += 1
                        
                        # Double-check result_frame is not empty
                        if result_frame is None or result_frame.size == 0:
                            # Return a fallback frame (black frame with text)
                            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(fallback, "Processing...", (20, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            return av.VideoFrame.from_ndarray(fallback, format="rgb24")
                            
                        return av.VideoFrame.from_ndarray(result_frame, format="rgb24")
                    except Exception as e:
                        # Return a fallback frame
                        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(fallback, f"Error: {str(e)[:30]}...", (20, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                        return av.VideoFrame.from_ndarray(fallback, format="rgb24")
            
            # Add columns for buttons
            col1, col2 = st.columns(2)
            
            # Add start/stop buttons for data collection
            with col1:
                if st.button("Start Data Collection"):
                    st.session_state.emotion_over_time = {emotion: [] for emotion in emotions}
                    st.session_state.frame_timestamps = []
                    st.session_state.should_plot = False
                    st.success("Data collection started! Use the webcam and emotions will be tracked.")
            
            with col2:
                if st.button("Stop and Generate Plots"):
                    st.session_state.should_plot = True
                    st.success("Data collection stopped. Generating plots...")
            
            # WebRTC streamer component with try/except
            try:
                webrtc_ctx = webrtc_streamer(
                    key="emotion-detection",
                    video_transformer_factory=EmotionDetector,
                    rtc_configuration=rtc_configuration
                )
                st.info("If the webcam appears black, please ensure you've granted camera permissions in your browser.")
                
                # Information about webcam usage
                if not webrtc_ctx.state.playing:
                    st.info("Click 'Start' button above to activate webcam")
                else:
                    st.info("Webcam is active! Click 'Stop' when done.")
            except Exception as e:
                st.error(f"WebRTC Error: {str(e)}")
                st.warning("Webcam functionality is not available in this environment. Please use the image upload feature instead.")
                webcam_available = False
            
            # Generate plots if requested
            if st.session_state.should_plot and len(st.session_state.frame_timestamps) > 0:
                st.subheader("Emotion Analysis Results")
                generate_plots()
            elif st.session_state.should_plot and len(st.session_state.frame_timestamps) == 0:
                st.warning("No emotion data collected. Please run the webcam first and ensure faces are detected.")
    
    except ImportError as e:
        st.error(f"Required module not available: {str(e)}")
        st.warning("WebRTC functionality cannot be loaded. Please use the image upload feature instead.")
        webcam_available = False
    except Exception as e:
        st.error(f"Unexpected error setting up webcam: {str(e)}")
        st.warning("WebRTC functionality cannot be loaded. Please use the image upload feature instead.")
        webcam_available = False

# If webcam is not available, provide a fallback
if detection_mode == "Webcam" and not webcam_available:
    st.info("Switching to image upload mode due to WebRTC issues...")
    st.warning("WebRTC can have issues on Streamlit Cloud. Try using the Upload Image mode instead.")
    
    # Create a simplified version of the upload image functionality
    st.subheader("Upload an image instead")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model and face detector
        model, device = load_model()
        face_detector = load_face_detector()
        gif_overlay = None  # Don't use GIF overlay in fallback mode
        
        if model is None or face_detector is None:
            st.error("Required components failed to load. Please check the error messages above.")
        else:
            try:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Process image
                result_img, faces = process_image(img, model, device, face_detector, gif_overlay)
                
                # Display result
                st.image(result_img, caption="Processed Image", use_container_width=True)
                
                if len(faces) == 0:
                    st.warning("No faces detected in the image!")
                else:
                    st.success(f"Detected {len(faces)} face(s) in the image!")
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")

# Add information about the project
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.write("""
This app uses a deep learning model to detect emotions in faces.
It can identify 8 different emotions:
- Neutral
- Happy 
- Surprise
- Sad
- Angry
- Disgust
- Fear
- Contempt
""")

# Add deployment instructions at the bottom
st.markdown("---")
st.markdown("### Project Structure")
st.code("""
emotion_recognition/
├── app.py                   # This Streamlit file
├── models.py                # Your model architecture file
├── gifoverlay.py            # Your GIF overlay handling code
├── model/
│   └── ferplus_model_pd_acc.pth  # Your trained model
└── EmojiGif/                # Folder with emoji GIF images
""")

# Add requirements.txt guidance
st.markdown("### Requirements")
st.write("Make sure you have a requirements.txt file with the following dependencies:")
st.code("""
streamlit==1.27.0
opencv-python-headless==4.8.0.76
torch==2.0.1
torchvision==0.15.2
matplotlib==3.7.2
numpy==1.24.3
streamlit-webrtc==0.47.1
av==10.0.0
aiortc==1.5.0
""")

st.markdown("---")
st.info("Note: If you encounter issues with the webcam functionality on Streamlit Cloud, try using the Upload Image mode instead.")