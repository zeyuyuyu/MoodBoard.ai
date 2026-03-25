import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import colorsys

class ImageAnalyzer:
    def __init__(self):
        self.emotion_weights = {
            'warm_colors': 0.4,
            'saturation': 0.3,
            'contrast': 0.3
        }

    def analyze_image(self, image_path: str) -> Dict:
        '''Analyze image for emotional intensity and color characteristics'''
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_array = np.array(img)
            
            results = {
                'emotion_score': self._calculate_emotion_score(img_array),
                'dominant_colors': self._extract_dominant_colors(img_array),
                'color_temperature': self._calculate_color_temperature(img_array),
                'intensity': self._calculate_intensity(img_array)
            }
            return results
        except Exception as e:
            raise Exception(f'Failed to analyze image: {str(e)}')

    def _calculate_emotion_score(self, img_array: np.ndarray) -> float:
        '''Calculate emotional intensity score based on color properties'''
        warm_color_score = self._calculate_warm_colors(img_array)
        saturation_score = self._calculate_saturation(img_array)
        contrast_score = self._calculate_contrast(img_array)
        
        emotion_score = (
            warm_color_score * self.emotion_weights['warm_colors'] +
            saturation_score * self.emotion_weights['saturation'] +
            contrast_score * self.emotion_weights['contrast']
        )
        return round(emotion_score, 2)

    def _extract_dominant_colors(self, img_array: np.ndarray, n_colors: int = 5) -> List[Tuple]:
        '''Extract dominant colors using k-means clustering'''
        pixels = img_array.reshape(-1, 3)
        pixels = pixels[::50]  # Sample pixels for performance
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        return [(int(r), int(g), int(b)) for r, g, b in colors]

    def _calculate_warm_colors(self, img_array: np.ndarray) -> float:
        '''Calculate ratio of warm colors in image'''
        hsv = self._rgb_to_hsv(img_array)
        hue = hsv[:,:,0]
        
        # Define warm colors as hues between 0-60 and 300-360 degrees
        warm_mask = ((hue >= 0) & (hue <= 60/360)) | (hue >= 300/360)
        return np.mean(warm_mask)

    def _calculate_saturation(self, img_array: np.ndarray) -> float:
        '''Calculate average saturation'''
        hsv = self._rgb_to_hsv(img_array)
        return np.mean(hsv[:,:,1])

    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        '''Calculate image contrast'''
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        return np.std(gray) / 128

    def _calculate_color_temperature(self, img_array: np.ndarray) -> float:
        '''Estimate color temperature (warm vs cool)'''
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        temperature = np.mean(r) / (np.mean(b) + 1e-6)
        return round(temperature, 2)

    def _calculate_intensity(self, img_array: np.ndarray) -> float:
        '''Calculate overall color intensity'''
        return round(np.mean(img_array) / 255, 2)

    def _rgb_to_hsv(self, rgb_array: np.ndarray) -> np.ndarray:
        '''Convert RGB array to HSV color space'''
        rgb_normalized = rgb_array / 255.0
        hsv = np.zeros_like(rgb_normalized)
        
        for i in range(rgb_array.shape[0]):
            for j in range(rgb_array.shape[1]):
                hsv[i,j] = colorsys.rgb_to_hsv(*rgb_normalized[i,j])
                
        return hsv