import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from models import PerformanceModel
from emotionoverlay import EmotionOverlay
from gifoverlay import GifEmotionOverlay
import time
from collections import deque

# Set page config
st.set_page_config(page_title="Real-time Facial Emotion Recognition", layout="wide")
st.title("Real-time Facial Emotion Recognition")

# Load model only once with optimization
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model/ferplus_model_pd_acc.pth"
    model = PerformanceModel(input_shape=(1, 48, 48), n_classes=8, logits=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Optimize model for inference
    if hasattr(torch.jit, 'script'):
        try:
            model = torch.jit.script(model)
        except:
            pass  # If scripting fails, use regular model
    
    return model, device

# Global variables
model, device = load_model()
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Simplified transform for better performance
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

gif_overlay = GifEmotionOverlay("EmojiGif/")

# Emotion color definitions (simplified)
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

# Performance settings in sidebar
st.sidebar.title("Performance Settings")
inference_frequency = st.sidebar.slider("Emotion Detection Frequency (frames)", 5, 30, 15, 
                                       help="Higher = less frequent detection = better performance")
face_detection_frequency = st.sidebar.slider("Face Detection Frequency (frames)", 10, 60, 30,
                                            help="Higher = less frequent face detection = better performance")
resolution_scale = st.sidebar.slider("Processing Resolution", 0.3, 0.8, 0.5, 0.1,
                                    help="Lower = faster processing")
max_faces = st.sidebar.slider("Max Faces to Process", 1, 5, 2,
                            help="Process fewer faces for better performance")
enable_gif_overlay = st.sidebar.checkbox("Enable GIF Overlay", False,
                                        help="Disable for significant performance boost")

class OptimizedVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.faces = []
        self.last_emotions = {}  # Cache emotions for each face
        self.face_tracking = {}  # Simple face tracking
        self.emotion_history = deque(maxlen=5)  # Smooth emotions
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        # Pre-allocate tensors for better performance
        self.face_tensor = torch.zeros((1, 1, 48, 48), dtype=torch.float32, device=device)
        
    def track_faces(self, new_faces):
        """Simple face tracking to maintain consistency"""
        if not hasattr(self, 'tracked_faces'):
            self.tracked_faces = {}
            self.next_face_id = 0
            
        current_faces = {}
        
        for face in new_faces:
            x, y, w, h = face
            center = (x + w//2, y + h//2)
            
            # Find closest existing face
            best_match = None
            min_distance = float('inf')
            
            for face_id, (old_center, old_face) in self.tracked_faces.items():
                distance = np.sqrt((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)
                if distance < min_distance and distance < 50:  # Threshold for matching
                    min_distance = distance
                    best_match = face_id
            
            if best_match is not None:
                current_faces[best_match] = (center, face)
            else:
                current_faces[self.next_face_id] = (center, face)
                self.next_face_id += 1
        
        self.tracked_faces = current_faces
        return current_faces
        
    def recv(self, frame):
        current_time = time.time()
        self.fps_counter.append(current_time - self.last_time)
        self.last_time = current_time
        
        img = frame.to_ndarray(format="bgr24")
        
        # Aggressive downscaling for processing
        original_h, original_w = img.shape[:2]
        process_w = int(original_w * resolution_scale)
        process_h = int(original_h * resolution_scale)
        img_small = cv2.resize(img, (process_w, process_h))
        
        # Convert to grayscale once
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Face detection (less frequent)
        if self.frame_count % face_detection_frequency == 0:
            detected_faces = haar_cascade.detectMultiScale(
                gray_small, 
                scaleFactor=1.3, 
                minNeighbors=3,  # Reduced for speed
                minSize=(30, 30)  # Smaller minimum size
            )
            
            # Limit number of faces and scale back coordinates
            if len(detected_faces) > max_faces:
                # Sort by size and take largest faces
                detected_faces = sorted(detected_faces, key=lambda f: f[2]*f[3], reverse=True)[:max_faces]
            
            # Scale coordinates back to original image size
            scale_factor = 1 / resolution_scale
            self.faces = [(int(x * scale_factor), int(y * scale_factor), 
                          int(w * scale_factor), int(h * scale_factor)) 
                         for (x, y, w, h) in detected_faces]
        
        # Track faces for consistency
        tracked = self.track_faces(self.faces)
        
        # Process emotions (even less frequent)
        if self.frame_count % inference_frequency == 0:
            for face_id, (center, (x, y, w, h)) in tracked.items():
                # Extract and preprocess face
                face_region = img[max(0, y):min(original_h, y+h), 
                                max(0, x):min(original_w, x+w)]
                
                if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
                    continue
                
                # Fast preprocessing
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                
                # Reuse pre-allocated tensor
                self.face_tensor[0, 0] = torch.from_numpy(face_resized).float().div(255).sub(0.5).div(0.5)
                
                # Model inference with no_grad for memory efficiency
                with torch.no_grad():
                    outputs = model(self.face_tensor)
                    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                    
                # Store emotion for this face
                self.last_emotions[face_id] = {
                    'probs': probs,
                    'top_emotion': emotions[np.argmax(probs)],
                    'confidence': float(np.max(probs))
                }
        
        # Render results
        for face_id, (center, (x, y, w, h)) in tracked.items():
            if face_id in self.last_emotions:
                emotion_data = self.last_emotions[face_id]
                top_emotion = emotion_data['top_emotion']
                confidence = emotion_data['confidence']
                
                # Draw bounding box
                color = emotion_colors.get(top_emotion, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Add GIF overlay only if enabled
                if enable_gif_overlay:
                    try:
                        img = gif_overlay.overlay_gif(img, top_emotion, x, y, w, h, 0)
                    except:
                        pass  # Skip overlay if it fails
                
                # Simple text display
                text = f"{top_emotion}: {int(confidence * 100)}%"
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
        
        # Display FPS
        if len(self.fps_counter) > 1:
            avg_frame_time = np.mean(self.fps_counter)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Optimized WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Lower resolution and frame rate for better performance
webrtc_ctx = webrtc_streamer(
    key="facial-emotion-optimized",
    video_processor_factory=OptimizedVideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480, "max": 640},  # Lower resolution
            "height": {"ideal": 360, "max": 480},
            "frameRate": {"ideal": 15, "max": 20}  # Lower frame rate
        }, 
        "audio": False
    },
    async_processing=True,
)

# Performance tips
with st.expander("🚀 Performance Tips"):
    st.write("""
    **For best performance:**
    1. **Disable GIF Overlay** - This provides the biggest performance boost
    2. **Increase Detection Frequencies** - Process emotions less often
    3. **Lower Resolution Scale** - Smaller images = faster processing
    4. **Limit Max Faces** - Process fewer faces simultaneously
    5. **Use a dedicated GPU server** - Streamlit Cloud has limited resources
    
    **If still too slow:**
    - Consider deploying on a cloud instance with GPU
    - Use a lighter model architecture
    - Implement frame skipping (process every 3rd frame)
    """)

with st.expander("About this optimized app"):
    st.write("""
    **Optimizations implemented:**
    - Reduced inference frequency (process emotions every 15 frames by default)
    - Aggressive image downscaling for processing
    - Face tracking to maintain consistency
    - Pre-allocated tensors to reduce memory allocation
    - Limited number of faces processed simultaneously
    - Optional GIF overlay (major performance impact when disabled)
    - Lower default video resolution and frame rate
    - Model optimization with TorchScript when available
    
    This app performs real-time facial emotion recognition with significant performance improvements
    over the original version.
    """)

# Add system info
st.sidebar.subheader("System Info")
st.sidebar.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if webrtc_ctx.video_processor:
    if hasattr(webrtc_ctx.video_processor, 'fps_counter') and len(webrtc_ctx.video_processor.fps_counter) > 1:
        avg_fps = 1.0 / np.mean(webrtc_ctx.video_processor.fps_counter)
        st.sidebar.metric("Current FPS", f"{avg_fps:.1f}")