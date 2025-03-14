
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models import PerformanceModel

# ✅ Load the trained PyTorch model (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "Your own Model Path"
model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Emotion categories (Order must match training data)
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# ✅ Define colors for each emotion
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

# ✅ OpenCV face detector
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
vid = cv2.VideoCapture(0)  # Start webcam

# ✅ Optimized Preprocessing Pipeline (No PIL conversion, Faster NumPy)
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize((48, 48)),  # Resize directly
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

print("✅ Model & Face Detector Loaded! Starting Webcam...")

frame_count = 0  # To control face detection frequency

while True:
    ret, frame = vid.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # ✅ Run face detection only every 3 frames (to reduce lag)
    if frame_count % 3 == 0:
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
    frame_count += 1

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]  # Extract face
        face_img = cv2.resize(face_img, (48, 48))  # Resize directly with OpenCV (faster than PIL)
        face_img = np.expand_dims(face_img, axis=0)  # Add channel dimension (1,48,48)
        face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)

        # ✅ Run model inference
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top_emotion_idx = np.argmax(probs)
            top_emotion = emotions[top_emotion_idx]

        # ✅ Get color for highest emotion
        top_color = emotion_colors[top_emotion]  

        # ✅ Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # ✅ Display emotions (Top emotion gets its unique color)
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            color = top_color if i == top_emotion_idx else (255, 255, 255)  # Unique color for top emotion, white for others
            text = f"{emotion}: {int(prob * 100)}%"
            cv2.putText(frame, text, (x, y - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()