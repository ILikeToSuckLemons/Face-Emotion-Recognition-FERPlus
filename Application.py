import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import time
import tempfile
import os
from PIL import Image

# Set page config first before any other Streamlit commands
st.set_page_config(page_title="Emotion Recognition", layout="wide")

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

# Simple EmotionOverlay class (simplified from your emotionoverlay.py)
class EmotionOverlay:
    def __init__(self):
        # Define emotion colors
        self.emotion_colors = {
            "Neutral": (255, 255, 255),  # White
            "Happy": (0, 255, 255),      # Yellow
            "Surprise": (0, 165, 255),   # Orange
            "Sad": (255, 0, 0),          # Blue
            "Angry": (0, 0, 255),        # Red
            "Disgust": (128, 0, 128),    # Purple
            "Fear": (255, 255, 0),       # Cyan
            "Contempt": (0, 255, 0)      # Green
        }
    
    def overlay(self, frame, emotion, x, y, w, h):
        # Draw rectangle with emotion color
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame

# Main app function
def main():
    st.title("Facial Emotion Recognition")
    
    # Sidebar for app controls
    st.sidebar.header("Settings")
    detection_frequency = st.sidebar.slider("Detection Frequency", 1, 10, 3, 
                                           help="How often to run face detection (frames)")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 
                                            help="Minimum confidence to display emotion")

    # App explanation
    with st.expander("About this app"):
        st.write("""
        This app detects facial emotions in real-time using a deep learning model.
        It can recognize 8 emotions: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, and Contempt.
        """)
    
    # Check if model exists in temp dir, if not explain upload
    model_upload_placeholder = st.empty()
    model_status = st.empty()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_status.info(f"Using device: {device}")
    
    # Model upload section
    uploaded_model = st.sidebar.file_uploader("Upload model file (ferplus_model_pd_acc.pth)", type=["pth"])
    
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path = tmp_file.name
            
        # Load model
        try:
            model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model_status.success("Model loaded successfully!")
        except Exception as e:
            model_status.error(f"Error loading model: {e}")
            return
    else:
        model_upload_placeholder.warning("Please upload your model file to continue")
        return
    
    # Emotion categories
    emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    
    # Initialize emotion overlay
    emotion_overlay = EmotionOverlay()
    
    # Transform for face preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Main app modes
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Webcam", "Upload Image"])
    
    if app_mode == "Webcam":
        # Webcam input
        st.subheader("Webcam Live Feed")
        
        run_webcam = st.checkbox("Start Webcam")
        
        # Placeholder for video frame and stats
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # OpenCV face detector
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        frame_count = 0
        start_time = time.time()
        faces = []
        
        # Webcam capture
        if run_webcam:
            vid = cv2.VideoCapture(0)
            
            if not vid.isOpened():
                st.error("Could not open webcam. Please check your camera settings.")
                return
                
            while run_webcam:
                ret, frame = vid.read()
                if not ret:
                    st.error("Failed to get frame from webcam")
                    break
                
                # Process every frame, but detect faces based on frequency setting
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if frame_count % detection_frequency == 0:
                    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
                
                frame_count += 1
                
                for (x, y, w, h) in faces:
                    # Extract and preprocess face
                    face_img = gray[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (48, 48))
                    
                    # Convert to tensor using transform
                    face_tensor = transform(face_img).unsqueeze(0).to(device)
                    
                    # Run model inference
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                        top_emotion_idx = np.argmax(probs)
                        top_emotion = emotions[top_emotion_idx]
                        top_prob = probs[top_emotion_idx]
                    
                    if top_prob >= confidence_threshold:
                        # Add overlay to the frame
                        frame = emotion_overlay.overlay(frame, top_emotion, x, y, w, h)
                        
                        # Display emotion text
                        text = f"{top_emotion}: {int(top_prob * 100)}%"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                
                # Add FPS counter to frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to RGB for Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                
                # Display stats
                stats_placeholder.text(f"FPS: {fps:.2f}, Frames processed: {frame_count}")
                
                # Add a small sleep to prevent high CPU usage
                time.sleep(0.01)
            
            # Release resources when stopped
            vid.release()
    
    elif app_mode == "Upload Image":
        st.subheader("Image Upload")
        
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
                
                # OpenCV face detector
                haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
                
                # Create a copy of the image for results
                result_image = image_bgr.copy()
                
                if len(faces) == 0:
                    st.warning("No faces detected in the image")
                else:
                    st.write(f"Found {len(faces)} face(s)")
                    
                    for (x, y, w, h) in faces:
                        # Extract and preprocess face
                        face_img = gray[y:y + h, x:x + w]
                        face_img = cv2.resize(face_img, (48, 48))
                        
                        # Convert to tensor
                        face_tensor = transform(face_img).unsqueeze(0).to(device)
                        
                        # Run model inference
                        with torch.no_grad():
                            outputs = model(face_tensor)
                            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                            
                        # Display all emotions and probabilities in a horizontal bar chart
                        emotion_results = {emotion: float(prob) for emotion, prob in zip(emotions, probs)}
                        emotion_df = {"Emotion": list(emotion_results.keys()), 
                                     "Probability": list(emotion_results.values())}
                        
                        # Get top emotion
                        top_emotion_idx = np.argmax(probs)
                        top_emotion = emotions[top_emotion_idx]
                        
                        # Add overlay to result image
                        result_image = emotion_overlay.overlay(result_image, top_emotion, x, y, w, h)
                        
                        # Add text with top emotion
                        text = f"{top_emotion}: {int(probs[top_emotion_idx] * 100)}%"
                        cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert back to RGB for display
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Display result image
                    st.image(result_rgb, caption="Detection Result", use_column_width=True)
                    
                    # Show emotion probabilities
                    st.subheader("Emotion Probabilities")
                    for emotion, prob in zip(emotions, probs):
                        st.write(f"{emotion}: {prob:.4f}")
                    
                    # Create a horizontal bar chart of emotions
                    emotion_data = [[emotion, float(prob)] for emotion, prob in zip(emotions, probs)]
                    emotion_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Use Streamlit's native chart
                    chart_data = {
                        "Emotion": [item[0] for item in emotion_data],
                        "Probability": [item[1] for item in emotion_data]
                    }
                    st.bar_chart(chart_data, x="Emotion", y="Probability")

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