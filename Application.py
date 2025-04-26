import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import os
from models import PerformanceModel
from emotionoverlay import EmotionOverlay
from gifoverlay import GifEmotionOverlay

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

@st.cache_resource
def load_model():
    # Load the trained PyTorch model (Use CPU for Streamlit Cloud)
    device = torch.device("cpu")
    model_path = "model/ferplus_model_pd_acc.pth"
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def main():
    st.title("Real-time Facial Emotion Recognition")
    
    # Initialize components
    model, device = load_model()
    gif_overlay = GifEmotionOverlay("EmojiGif/")
    
    # Emotion categories (Order must match training data)
    emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    
    # OpenCV face detector
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Optimized Preprocessing Pipeline 
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Resize((48, 48)),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    
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
    
    # Initialize variables for animation
    color_index = 0
    frame_count = 0
    animation_offset = 0
    offset_direction = 1
    
    # Streamlit camera input
    camera_image = st.camera_input("Take a picture or enable camera")
    
    if camera_image is not None:
        # Convert from Streamlit image to OpenCV format
        bytes_data = camera_image.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Process the frame (similar to original code)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Change color every 10 frames
        if frame_count % 10 == 0:  
            color_index = (color_index + 1) % 3
            
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
        frame_count += 1
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.expand_dims(face_img, axis=0)
            face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)
            
            # Run model inference
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_emotion_idx = np.argmax(probs)
                top_emotion = emotions[top_emotion_idx]
            
            # Floating animation logic
            animation_offset += offset_direction * 2
            if abs(animation_offset) > 10:
                offset_direction *= -1
            
            # Overlay character
            frame = gif_overlay.overlay_gif(frame, top_emotion, x, y, w, h, animation_offset)
            
            # Draw rectangle around face
            box_color = emotion_colors.get(top_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Display emotions
            for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                if i == top_emotion_idx:
                    text_color = emotion_text_colors[top_emotion][color_index]
                else:
                    text_color = (255, 255, 255)
                
                text = f"{emotion}: {int(prob * 100)}%"
                cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Convert back to RGB for display (Streamlit uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Processed Image", use_column_width=True)
        
        # Display emotion probabilities as a bar chart
        if len(faces) > 0:
            st.write("Emotion Probabilities:")
            chart_data = {emotion: float(prob) for emotion, prob in zip(emotions, probs)}
            st.bar_chart(chart_data)

if __name__ == "__main__":
    main()