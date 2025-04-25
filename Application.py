import streamlit as st
try:
    import cv2
except ImportError:
    import sys
    !{sys.executable} -m pip install opencv-python-headless==4.9.0.80
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
    model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
    model.eval()
    return model, device

# Load GIF emotion overlay
@st.cache_resource
def load_gif_overlay():
    return GifEmotionOverlay("EmojiGif/")

# Get face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion prediction function
def predict_emotion(face_tensor, model):
    with torch.no_grad():
        outputs = model(face_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        top_emotion_idx = np.argmax(probs)
        top_emotion = emotions[top_emotion_idx]
    return probs, top_emotion, top_emotion_idx

# Process image function - optimized version
def process_image(img, model, device, face_detector, gif_overlay=None):
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
            img_with_overlay = gif_overlay.overlay_gif(
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 
                top_emotion, x, y, w, h, animation_offset
            )
            img_rgb = cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB)
        
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
    else:
        st.warning("No emotion data collected yet. Please run the webcam first.")

# Main app logic based on selection
if detection_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model and face detector
        model, device = load_model()
        face_detector = load_face_detector()
        gif_overlay = load_gif_overlay()
        
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

elif detection_mode == "Webcam":
    # Load model and face detector
    model, device = load_model()
    face_detector = load_face_detector()
    gif_overlay = load_gif_overlay()
    
    # Create webcam container
    webcam_container = st.container()
    
    # Add columns for buttons
    col1, col2 = st.columns(2)
    
    # Add start/stop buttons
    with col1:
        start_button = st.button("Start Webcam")
    with col2:
        stop_button = st.button("Stop Webcam")
    
    # Handle button states
    if start_button:
        st.session_state.run_webcam = True
        st.session_state.should_plot = False
        # Clear previous data when starting new session
        st.session_state.emotion_over_time = {emotion: [] for emotion in emotions}
        st.session_state.frame_timestamps = []
        st.session_state.color_index = 0
        st.session_state.frame_count = 0
    
    if stop_button:
        st.session_state.run_webcam = False
        st.session_state.should_plot = True
    
    # Placeholder for webcam feed
    frame_window = webcam_container.empty()
    
    # Run webcam if enabled
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam!")
        else:
            st.info("Webcam is active! Click 'Stop Webcam' when done.")
            
            # Use a more efficient webcam processing loop
            try:
                while st.session_state.run_webcam:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame from webcam!")
                        break
                    
                    # Process frame based on counter for performance
                    if st.session_state.frame_count % st.session_state.process_every_n_frames == 0:
                        # Update animation offset (simpler pattern)
                        st.session_state.animation_offset = (st.session_state.animation_offset + 1) % 10
                        
                        # Process frame - full processing
                        result_frame, faces = process_image(frame, model, device, face_detector, gif_overlay)
                        
                        # Update emotion data if faces found
                        if faces:
                            update_emotion_data(faces)
                    else:
                        # For skipped frames, just display with previous detection
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result_frame = img_rgb  # Simple display for skipped frames
                    
                    # Display the frame
                    frame_window.image(result_frame, channels="RGB", use_container_width=True)
                    
                    # Increment frame counter
                    st.session_state.frame_count += 1
                    
                    # Very minimal sleep - just enough to prevent UI freezing
                    time.sleep(0.03)  # Reduced from 0.1
            finally:
                # Always release the webcam when done
                cap.release()
    
    # Generate plots after webcam stops
    if st.session_state.should_plot:
        st.subheader("Emotion Analysis Results")
        generate_plots()
    elif not st.session_state.run_webcam:
        frame_window.info("Click 'Start Webcam' to begin face emotion detection.")

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
