import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models import PerformanceModel
from emotionoverlay import EmotionOverlay  # If you still want the static overlay
from gifoverlay import GifEmotionOverlay
from PIL import Image
import tempfile

# Load model and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/ferplus_model_pd_acc.pth"
model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize GifOverlay
gif_overlay = GifEmotionOverlay("EmojiGif/")

# Colors setup
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
    "Neutral": (255, 255, 255),
    "Happy": (0, 255, 255),
    "Surprise": (0, 165, 255),
    "Sad": (255, 0, 0),
    "Angry": (0, 0, 255),
    "Disgust": (128, 0, 128),
    "Fear": (255, 255, 0),
    "Contempt": (0, 255, 0)
}

# Streamlit app layout
st.title("Real-Time Facial Emotion Recognition")

color_index = 0
frame_count = 0
animation_offset = 0
offset_direction = 1

# Camera input from user
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Read image
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV BGR

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Change color every 10 frames
    if frame_count % 10 == 0:
        color_index = (color_index + 1) % 3

    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    frame_count += 1

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(face_img, axis=0)
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]

        # Floating animation
        animation_offset += offset_direction * 2
        if abs(animation_offset) > 10:
            offset_direction *= -1

        # Overlay gif
        frame = gif_overlay.overlay_gif(frame, top_emotion, x, y, w, h, animation_offset)

        # Draw rectangle
        box_color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Emotion labels
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            if i == top_emotion_idx:
                text_color = emotion_text_colors[top_emotion][color_index]
            else:
                text_color = (255, 255, 255)
            text = f"{emotion}: {int(prob * 100)}%"
            cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Convert frame back to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB")
