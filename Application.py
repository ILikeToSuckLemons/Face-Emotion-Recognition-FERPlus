import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models import PerformanceModel
from gifoverlay import GifEmotionOverlay
import cv2

# Initialize GIF Overlay
gif_overlay = GifEmotionOverlay("EmojiGif/")  # Use the correct folder

# Load the trained PyTorch model (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/ferplus_model_pd_acc.pth"
model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Emotion categories (Order must match training data)
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# OpenCV face detector
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Optimized Preprocessing Pipeline (No PIL conversion, Faster NumPy)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),  # Resize directly
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Emotion text and box color mapping
emotion_colors = {
    "Neutral": (255, 255, 255),  # White
    "Happy": (0, 255, 255),  # Yellow
    "Surprise": (0, 165, 255),  # Orange
    "Sad": (255, 0, 0),  # Blue
    "Angry": (0, 0, 255),  # Red
    "Disgust": (128, 0, 128),  # Purple
    "Fear": (255, 255, 0),  # Cyan
    "Contempt": (0, 255, 0)  # Green
}

# Streamlit Camera Input for live webcam feed
st.title("Real-Time Facial Emotion Recognition")
camera_input = st.camera_input("Capture", key="camera")

if camera_input:
    frame_count = 0  # Frame counter for selective processing
    animation_offset = 0  # For floating animation
    offset_direction = 1  # To make it go up/down
    
    # Start webcam loop (using Streamlit's real-time image handling)
    while True:
        # Get the webcam frame
        frame = camera_input

        if frame is not None:
            # Convert Streamlit image to OpenCV format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # Run face detection only every 3 frames to reduce lag
            if frame_count % 3 == 0:
                faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
            frame_count += 1

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

                # Floating animation logic
                animation_offset += offset_direction * 2  # Move up/down by 2 pixels
                if abs(animation_offset) > 10:
                    offset_direction *= -1  # Reverse direction

                # Overlay character
                frame = gif_overlay.overlay_gif(frame, top_emotion, x, y, w, h, animation_offset)

                # Draw rectangle around face
                box_color = emotion_colors.get(top_emotion, (255, 255, 255))  # Default to white if not found
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)  # Draw box with emotion color

                # Display emotions
                for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                    # Use unique color for the top emotion
                    if i == top_emotion_idx:
                        text_color = (255, 255, 255)  # White for top emotion
                    else:
                        text_color = (0, 0, 0)  # Default black for others

                    text = f"{emotion}: {int(prob * 100)}%"
                    cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Display the frame in Streamlit
            st.image(frame, channels="BGR", use_column_width=True)

        # Break out of the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
