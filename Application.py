import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from models import PerformanceModel
import time
from collections import deque

# GPU Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
    st.success(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    torch.set_num_threads(4)  # Optimize CPU threads
    st.warning("⚠️ GPU not available, using CPU")

st.set_page_config(page_title="GPU-Optimized Emotion Recognition", layout="wide")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_model():
    """Load and optimize the emotion recognition model"""
    try:
        model_path = "model/ferplus_model_pd_acc.pth"
        model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Optimize model for inference
        if device.type == "cuda":
            # GPU optimization
            model = model.half()  # Use FP16 for faster inference
            dummy_input = torch.randn(1, 1, 48, 48, device=device, dtype=torch.half)
        else:
            # CPU optimization
            dummy_input = torch.randn(1, 1, 48, 48, device=device)
        
        # Trace the model for optimization
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.eval()
        
        st.success("✅ Model loaded successfully!")
        return traced_model, device.type == "cuda"
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None, False

model, is_gpu = load_model()
if model is None:
    st.stop()

# Emotion labels and colors
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
emotion_colors = {
    "Neutral": (128, 128, 128),
    "Happy": (0, 255, 255),      # Yellow
    "Surprise": (0, 165, 255),   # Orange
    "Sad": (255, 0, 0),          # Blue
    "Angry": (0, 0, 255),        # Red
    "Disgust": (128, 0, 128),    # Purple
    "Fear": (255, 255, 0),       # Cyan
    "Contempt": (0, 255, 0)      # Green
}

# Performance settings based on hardware
if is_gpu:
    # GPU settings - more aggressive processing
    INFERENCE_EVERY_N_FRAMES = 3      # Process every 3 frames
    FACE_DETECTION_EVERY_N_FRAMES = 10  # Detect faces every 10 frames
    PROCESSING_SCALE = 0.5            # Process at 50% resolution
    BATCH_SIZE = 4                    # Process multiple faces if available
else:
    # CPU settings - conservative processing
    INFERENCE_EVERY_N_FRAMES = 15     # Process every 15 frames
    FACE_DETECTION_EVERY_N_FRAMES = 30  # Detect faces every 30 frames
    PROCESSING_SCALE = 0.3            # Process at 30% resolution
    BATCH_SIZE = 1                    # Process one face at a time

class OptimizedEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.faces = []
        self.current_emotions = {}  # Store emotions for multiple faces
        self.last_inference_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Pre-allocate tensors for efficiency
        if is_gpu:
            self.face_tensor = torch.zeros(
                (BATCH_SIZE, 1, 48, 48), 
                device=device, 
                dtype=torch.half
            )
        else:
            self.face_tensor = torch.zeros(
                (BATCH_SIZE, 1, 48, 48), 
                device=device, 
                dtype=torch.float32
            )
    
    def detect_faces(self, gray_img):
        """Detect faces in the image"""
        faces = self.face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        return faces
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        # Resize to 48x48
        face_resized = cv2.resize(face_img, (48, 48))
        
        # Normalize to [-1, 1]
        face_normalized = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        return face_normalized
    
    def predict_emotion(self, face_regions):
        """Predict emotions for face regions"""
        if not face_regions:
            return []
        
        batch_size = min(len(face_regions), BATCH_SIZE)
        
        # Prepare batch
        for i, face_region in enumerate(face_regions[:batch_size]):
            face_processed = self.preprocess_face(face_region)
            self.face_tensor[i, 0] = torch.from_numpy(face_processed).to(
                device=device, 
                dtype=torch.half if is_gpu else torch.float32
            )
        
        # Inference
        with torch.no_grad():
            if batch_size < BATCH_SIZE:
                outputs = model(self.face_tensor[:batch_size])
            else:
                outputs = model(self.face_tensor)
            
            probs = F.softmax(outputs, dim=1)
            
        # Convert to CPU for processing
        probs_cpu = probs.cpu().numpy()
        
        results = []
        for i in range(batch_size):
            emotion_idx = np.argmax(probs_cpu[i])
            confidence = float(probs_cpu[i][emotion_idx])
            emotion = emotions[emotion_idx]
            results.append((emotion, confidence))
        
        return results
    
    def recv(self, frame):
        frame_start_time = time.time()
        
        # Convert frame
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Resize for processing
        small_w, small_h = int(w * PROCESSING_SCALE), int(h * PROCESSING_SCALE)
        img_small = cv2.resize(img, (small_w, small_h))
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        if self.frame_count % FACE_DETECTION_EVERY_N_FRAMES == 0:
            faces_small = self.detect_faces(gray_small)
            
            # Scale faces back to original size
            scale_factor = 1.0 / PROCESSING_SCALE
            self.faces = []
            for (x, y, w_face, h_face) in faces_small:
                x_orig = int(x * scale_factor)
                y_orig = int(y * scale_factor)
                w_orig = int(w_face * scale_factor)
                h_orig = int(h_face * scale_factor)
                
                # Ensure face is within image bounds
                x_orig = max(0, min(x_orig, w - 1))
                y_orig = max(0, min(y_orig, h - 1))
                w_orig = min(w_orig, w - x_orig)
                h_orig = min(h_orig, h - y_orig)
                
                if w_orig > 20 and h_orig > 20:  # Minimum face size
                    self.faces.append((x_orig, y_orig, w_orig, h_orig))
        
        # Emotion inference
        if (self.frame_count % INFERENCE_EVERY_N_FRAMES == 0 and 
            len(self.faces) > 0):
            
            # Extract face regions
            face_regions = []
            valid_faces = []
            
            for face_coords in self.faces:
                x, y, w_face, h_face = face_coords
                
                # Extract face region
                y_end = min(y + h_face, h)
                x_end = min(x + w_face, w)
                
                if y < h and x < w and y_end > y and x_end > x:
                    face_region = img[y:y_end, x:x_end]
                    if face_region.size > 0:
                        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                        face_regions.append(face_gray)
                        valid_faces.append(face_coords)
            
            # Predict emotions
            if face_regions:
                emotion_results = self.predict_emotion(face_regions)
                
                # Update emotion cache
                self.current_emotions = {}
                for i, (emotion, confidence) in enumerate(emotion_results):
                    if i < len(valid_faces):
                        face_id = f"face_{i}"
                        self.current_emotions[face_id] = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'coords': valid_faces[i]
                        }
        
        # Draw results
        for face_id, emotion_data in self.current_emotions.items():
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            x, y, w_face, h_face = emotion_data['coords']
            
            # Get color
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w_face, y + h_face), color, 2)
            
            # Draw emotion label
            label = f"{emotion} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(img, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(img, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # FPS calculation
        frame_time = time.time() - frame_start_time
        self.fps_history.append(frame_time)
        
        if len(self.fps_history) > 5:
            avg_frame_time = np.mean(list(self.fps_history)[-10:])
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Draw FPS
            fps_text = f"FPS: {fps:.1f} | {device.type.upper()}"
            cv2.putText(img, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face count
        face_count_text = f"Faces: {len(self.current_emotions)}"
        cv2.putText(img, face_count_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Configuration
rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Video constraints based on hardware capability
if is_gpu:
    video_constraints = {
        "width": {"ideal": 640, "max": 1280},
        "height": {"ideal": 480, "max": 720},
        "frameRate": {"ideal": 30, "max": 60}
    }
else:
    video_constraints = {
        "width": {"ideal": 320, "max": 640},
        "height": {"ideal": 240, "max": 480},
        "frameRate": {"ideal": 15, "max": 30}
    }

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=OptimizedEmotionProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": video_constraints,
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.markdown("### 🎭 Emotion Legend")
    for emotion, color in emotion_colors.items():
        # Convert BGR to RGB for display
        rgb_color = f"rgb({color[2]}, {color[1]}, {color[0]})"
        st.markdown(
            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
            f'<div style="width: 20px; height: 20px; background-color: {rgb_color}; '
            f'margin-right: 10px; border-radius: 3px;"></div>'
            f'<span>{emotion}</span></div>',
            unsafe_allow_html=True
        )
    
    # Performance info
    st.markdown("### ⚡ Performance Settings")
    if is_gpu:
        st.success("🚀 GPU Accelerated")
        st.info(f"• Inference every {INFERENCE_EVERY_N_FRAMES} frames\n"
                f"• Face detection every {FACE_DETECTION_EVERY_N_FRAMES} frames\n"
                f"• Processing scale: {PROCESSING_SCALE}x\n"
                f"• Batch size: {BATCH_SIZE}")
    else:
        st.warning("💻 CPU Mode")
        st.info(f"• Inference every {INFERENCE_EVERY_N_FRAMES} frames\n"
                f"• Face detection every {FACE_DETECTION_EVERY_N_FRAMES} frames\n"
                f"• Processing scale: {PROCESSING_SCALE}x")

# Instructions
st.markdown("---")
st.markdown("""
### 📝 Instructions
1. **Allow camera access** when prompted
2. **Position your face** within the camera view
3. **Multiple faces** will be detected and analyzed simultaneously
4. **Emotions update** in real-time with confidence scores

### 🔧 Troubleshooting
- If you experience lag, the app will automatically adjust processing frequency
- For best results, ensure good lighting and clear face visibility
- GPU acceleration provides significantly better performance
""")

# Requirements info
with st.expander("📋 System Requirements"):
    st.markdown("""
    **For optimal performance:**
    - NVIDIA GPU with CUDA support (recommended)
    - At least 4GB RAM
    - Modern web browser with WebRTC support
    
    **Dependencies:**
    ```
    streamlit
    streamlit-webrtc
    opencv-python
    torch
    torchvision
    numpy
    av
    ```
    """)