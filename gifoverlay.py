import cv2
import os
from PIL import Image
import numpy as np

class GifEmotionOverlay:
    def __init__(self, gif_dir, scale_factor=0.3):
        self.gif_dir = gif_dir
        self.scale_factor = scale_factor
        self.gif_frames = {}  # Dict of emotion -> list of frames
        self.frame_index = {}  # Dict of emotion -> current frame index
        self.load_gifs()

    def load_gifs(self):
        emotions = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise", "Neutral", "Contempt"]
        for emotion in emotions:
            path = os.path.join(self.gif_dir, f"{emotion}.gif")
            if os.path.exists(path):
                gif = Image.open(path)
                frames = []

                try:
                    while True:
                        gif_rgba = gif.convert("RGBA")
                        frame = np.array(gif_rgba)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)  # Convert to OpenCV format
                        frame = self.resize_image(frame)
                        frames.append(frame)
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass  # End of GIF

                if frames:
                    self.gif_frames[emotion] = frames
                    self.frame_index[emotion] = 0  # Start at first frame

    def resize_image(self, img):
        h, w = img.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def overlay_gif(self, frame, emotion, face_x, face_y, face_w, face_h, animation_offset=0):
        if emotion not in self.gif_frames:
            return frame

        frames = self.gif_frames[emotion]
        index = self.frame_index[emotion]

        overlay = frames[index]
        self.frame_index[emotion] = (index + 1) % len(frames)  # Loop animation

        oh, ow = overlay.shape[:2]

        # Position to the right of the face
        x_offset = face_x + face_w + 10
        y_offset = face_y + (face_h // 4) + animation_offset

        # Clip to frame bounds
        x_offset = min(x_offset, frame.shape[1] - ow)
        y_offset = max(0, min(y_offset, frame.shape[0] - oh))

        return self.overlay_transparent(frame, overlay, x_offset, y_offset)

    def overlay_transparent(self, background, overlay, x, y):
        bh, bw = background.shape[:2]
        h, w = overlay.shape[:2]

        if x + w > bw or y + h > bh:
            return background

        overlay_img = overlay[:, :, :3]
        mask = overlay[:, :, 3:]  # Alpha channel

        roi = background[y:y+h, x:x+w]

        background[y:y+h, x:x+w] = roi * (1 - mask / 255) + overlay_img * (mask / 255)
        return background
