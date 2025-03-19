import cv2
import numpy as np
import os

class EmotionOverlay:
    def __init__(self, image_dir, scale_factor=0.3):
        """Loads and prepares emotion images."""
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.emotion_images = {}
        self.load_images()
    
    def load_images(self):
        """Loads emotion images from directory and resizes them."""
        emotions = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise", "Neutral", "Contempt"]
        for emotion in emotions:
            img_path = os.path.join(self.image_dir, f"{emotion.lower()}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load with transparency
                if img is not None:
                    img = self.resize_image(img)
                    self.emotion_images[emotion] = img
    
    def resize_image(self, img):
        """Resizes an image while maintaining aspect ratio."""
        h, w = img.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def overlay_image(self, frame, emotion, face_x, face_y, face_w, face_h, animation_offset=0):
        """Overlays the emotion image near the detected face with an optional floating effect."""
        if emotion not in self.emotion_images:
            return frame  # No matching image, return original frame
        
        overlay = self.emotion_images[emotion]
        oh, ow = overlay.shape[:2]
        
        # Position the image to the right of the detected face
        x_offset = face_x + face_w + 10  # Move right of the face box
        y_offset = face_y + (face_h // 4) + animation_offset  # Align at face level

        # Ensure it stays within frame bounds
        x_offset = min(x_offset, frame.shape[1] - ow)  # Prevent going off-screen
        y_offset = max(0, min(y_offset, frame.shape[0] - oh))  # Keep within height

        
        # Overlay the image with transparency
        return self.overlay_transparent(frame, overlay, x_offset, y_offset)
    
    def overlay_transparent(self, background, overlay, x, y):
        """Overlays transparent images onto the frame."""
        bh, bw = background.shape[:2]
        h, w = overlay.shape[:2]
        
        if x + w > bw or y + h > bh:
            return background  # Avoid out-of-bounds error
        
        overlay_img = overlay[:, :, :3]  # RGB
        mask = overlay[:, :, 3:]  # Alpha
        
        roi = background[y:y+h, x:x+w]
        
        # Blend the images based on transparency mask
        background[y:y+h, x:x+w] = roi * (1 - mask / 255) + overlay_img * (mask / 255)
        return background
