import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st
from models import PerformanceModel
from emotionoverlay import EmotionOverlay
from gifoverlay import GifEmotionOverlay

# Initialize GifOverlay
gif_overlay = GifEmotionOverlay("EmojiGif/")

# Load the trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/ferplus_model_pd_acc.pth"
model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Emotion categories (Order must match training data)
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Optimized Preprocessing Pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Emotion colors
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

# Streamlit camera input
st.title("Real-Time Emotion Recognition")
camera_input = st.camera_input("Capture Webcam Feed")

if camera_input:
    # Capture the image from Streamlit's camera input (as bytes)
    frame = np.frombuffer(camera_input.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OpenCV face detector
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]  # Extract face
        face_img = cv2.resize(face_img, (48, 48))  # Resize directly with OpenCV (faster than PIL)
        face_img = np.expand_dims(face_img, axis=0)  # Add channel dimension (1,48,48)
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)

        # Run model inference
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]

        # Overlay GIF on face region
        frame = gif_overlay.overlay_gif(frame, top_emotion, x, y, w, h)

        # Draw rectangle around face
        box_color = emotion_colors.get(top_emotion, (255, 255, 255))  # Default to white if not found
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)  # Draw box with emotion color

        # Display emotions
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            text_color = (255, 255, 255)  # Default text color white
            text = f"{emotion}: {int(prob * 100)}%"
            cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Convert back to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display frame in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)
