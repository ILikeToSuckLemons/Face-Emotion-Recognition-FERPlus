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
from emotionoverlay import EmotionOverlay

# App title and description
st.title("Facial Emotion Recognition")
st.write("Upload an image or use your webcam to detect emotions in real-time!")

# Sidebar for app controls
st.sidebar.header("Settings")
detection_mode = st.sidebar.radio("Choose Detection Mode", ["Upload Image", "Webcam"])

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

# Define emotion colors
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

# Define emotion text colors
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

# Load model function - @st.cache_resource prevents reloading on every rerun
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load("model/ferplus_model_pd_acc.pth", map_location=device))
    model.eval()
    return model, device

# Load emotion overlay
@st.cache_resource
def load_emotion_overlay():
    return EmotionOverlay("emojiImages/")

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

# Process image function
def process_image(img, model, device, face_detector, overlay, animation_offset=0):
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        probs, top_emotion, top_emotion_idx = predict_emotion(face_tensor, model)
        
        # Apply emotion overlay
        img_rgb = cv2.cvtColor(overlay.overlay_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 
                                                   top_emotion, x, y, w, h, animation_offset), 
                              cv2.COLOR_BGR2RGB)
        
        # Draw rectangle around face
        color = emotion_colors.get(top_emotion, (255, 255, 255))
        color_rgb = (color[2], color[1], color[0])  # Convert BGR to RGB
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color_rgb, 2)
        
        # Use color cycling for the top emotion
        color_index = int(time.time() * 0.1) % 3
        
        # Display emotion probabilities
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            # Choose text color
            if i == top_emotion_idx:
                text_color = emotion_text_colors[top_emotion][color_index]
                # Convert BGR to RGB
                text_color = (text_color[2], text_color[1], text_color[0])
            else:
                text_color = (255, 255, 255)
                
            text = f"{emotion}: {int(prob * 100)}%"
            cv2.putText(img_rgb, text, (x, y-10-(i*20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return img_rgb, faces

# Create function to update emotion over time
def update_emotion_data(frame, model, device, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    
    if len(faces_detected) > 0:
        (x, y, w, h) = faces_detected[0]  # Just take the first face for tracking
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        for i, emotion in enumerate(emotions):
            st.session_state.emotion_over_time[emotion].append(probs[i])

        st.session_state.frame_timestamps.append(len(st.session_state.frame_timestamps))

# Function to generate and display plots
def generate_plots():
    # Only generate plots if we have data
    if len(st.session_state.frame_timestamps) > 0:
        # Plot 1: Total Emotion Scores
        emotion_totals = {
            emotion: sum(st.session_state.emotion_over_time[emotion])
            for emotion in emotions
        }

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(emotion_totals.keys(), emotion_totals.values(), color='skyblue')
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
            ax2.plot(st.session_state.frame_timestamps[:len(scores)], scores, label=emotion, linewidth=1.5)

        ax2.set_title("Emotion Scores Over Time")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Probability")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("No emotion data collected yet. Please run the webcam first.")

# Main app logic based on selection
if detection_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model and face detector
        model, device = load_model()
        face_detector = load_face_detector()
        overlay = load_emotion_overlay()
        
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process image
        result_img, faces = process_image(img, model, device, face_detector, overlay)
        
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
    overlay = load_emotion_overlay()
    
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
    
    if stop_button:
        st.session_state.run_webcam = False
        st.session_state.should_plot = True
    
    # Placeholder for webcam feed
    webcam_placeholder = webcam_container.empty()
    
    # Run webcam if enabled
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam!")
        else:
            st.info("Webcam is active! Click 'Stop Webcam' when done.")
            
            # Use a loop with short sleep to update the frame
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame from webcam!")
                    break
                
                # Update animation offset
                st.session_state.animation_offset += st.session_state.offset_direction * 2
                if abs(st.session_state.animation_offset) > 10:
                    st.session_state.offset_direction *= -1
                
                # Process frame
                result_frame, faces = process_image(frame, model, device, face_detector, 
                                                   overlay, st.session_state.animation_offset)
                
                # Update emotion data
                update_emotion_data(frame, model, device, face_detector)
                
                # Display frame - Fix for deprecated parameter
                webcam_placeholder.image(result_frame, channels="RGB", use_container_width=True)
                
                # Short delay to prevent UI freezing
                time.sleep(0.1)
            
            # Release the webcam when stopped
            cap.release()
    
    # Generate plots after webcam stops
    if st.session_state.should_plot:
        st.subheader("Emotion Analysis Results")
        generate_plots()
    elif not st.session_state.run_webcam:
        webcam_placeholder.info("Click 'Start Webcam' to begin face emotion detection.")

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
├── emotionoverlay.py        # Your overlay handling code
├── model/
│   └── ferplus_model_pd_acc.pth  # Your trained model
└── emojiImages/             # Folder with emoji images
""")