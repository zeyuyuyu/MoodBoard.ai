import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class ImageAnalyzer:
    def __init__(self):
        self.emotion_model = load_model('models/emotion_model.h5')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image for emotional content and color mood
        Returns dictionary with emotion predictions and color analysis
        """
        # Load and process image
        img = Image.open(image_path)
        cv_img = cv2.imread(image_path)
        
        results = {
            'emotions': self._detect_emotions(cv_img),
            'color_mood': self._analyze_colors(img),
            'dominant_colors': self._get_dominant_colors(img)
        }
        return results

    def _detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """Detect faces and predict emotions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions_detected = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = self.emotion_model.predict(roi)[0]
            emotion = {
                'emotion': self.emotions[np.argmax(preds)],
                'confidence': float(np.max(preds)),
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            }
            emotions_detected.append(emotion)
            
        return emotions_detected

    def _analyze_colors(self, image: Image.Image) -> Dict:
        """Analyze color composition for mood"""
        # Convert to RGB and get color data
        img_rgb = image.convert('RGB')
        pixels = np.float32(img_rgb).reshape(-1, 3)
        
        # Calculate average colors
        avg_colors = np.mean(pixels, axis=0)
        
        # Calculate color mood based on psychology
        r, g, b = avg_colors
        
        # Simple mood mapping based on color psychology
        mood = {
            'energy': min(100, (r * 0.8 + b * 0.2) / 2.55),
            'calmness': min(100, (b * 0.6 + g * 0.4) / 2.55),
            'warmth': min(100, (r * 0.6 + g * 0.4) / 2.55),
            'average_rgb': {'r': int(r), 'g': int(g), 'b': int(b)}
        }
        
        return mood

    def _get_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple]:
        """Extract dominant colors using k-means clustering"""
        # Resize image to speed up processing
        img = image.copy()
        img.thumbnail((150, 150))
        
        # Get colors from image
        pixels = np.float32(img).reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        
        # Convert colors to RGB tuples
        dominant_colors = []
        for color in palette:
            dominant_colors.append(tuple(map(int, color)))
            
        return dominant_colors