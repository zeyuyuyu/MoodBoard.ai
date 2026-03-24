# Image color mood analysis for MoodBoard.ai
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

class MoodAnalyzer:
    def __init__(self):
        self.mood_colors = {
            'energetic': [(255,0,0), (255,165,0), (255,255,0)],  # Reds, oranges, yellows
            'calm': [(0,255,255), (135,206,235), (173,216,230)], # Blues, light blues
            'peaceful': [(144,238,144), (152,251,152), (143,188,143)], # Soft greens
            'romantic': [(255,192,203), (255,182,193), (255,20,147)],  # Pinks
            'mysterious': [(75,0,130), (72,61,139), (106,90,205)]      # Deep purples
        }

    def extract_dominant_colors(self, image_path, n_colors=5):
        """Extract dominant color palette from image using K-means clustering"""
        img = Image.open(image_path)
        img = img.resize((150, 150))  # Resize for performance
        img_array = np.array(img)
        
        # Reshape the array for K-means
        pixels = img_array.reshape((-1, 3))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Get the colors and their percentages
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate color percentages
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Convert to integer RGB values and pair with percentages
        palette = [(tuple(map(int, color)), pct) 
                  for color, pct in zip(colors, percentages)]
        
        return sorted(palette, key=lambda x: x[1], reverse=True)

    def calculate_color_distance(self, color1, color2):
        """Calculate perceptual distance between two colors using Lab color space"""
        rgb1 = sRGBColor(color1[0]/255, color1[1]/255, color1[2]/255)
        rgb2 = sRGBColor(color2[0]/255, color2[1]/255, color2[2]/255)
        
        lab1 = convert_color(rgb1, LabColor)
        lab2 = convert_color(rgb2, LabColor)
        
        delta_e = np.sqrt(
            (lab1.lab_l - lab2.lab_l) ** 2 +
            (lab1.lab_a - lab2.lab_a) ** 2 +
            (lab1.lab_b - lab2.lab_b) ** 2
        )
        return delta_e

    def analyze_mood(self, image_path):
        """Analyze the mood of an image based on its color palette"""
        dominant_colors = self.extract_dominant_colors(image_path)
        
        mood_scores = {mood: 0 for mood in self.mood_colors.keys()}
        
        # For each dominant color, calculate its influence on different moods
        for dom_color, percentage in dominant_colors:
            for mood, mood_palette in self.mood_colors.items():
                # Find minimum distance to any color in the mood's palette
                min_distance = min(
                    self.calculate_color_distance(dom_color, mood_color)
                    for mood_color in mood_palette
                )
                # Convert distance to similarity score (inverse relationship)
                similarity = 1 / (1 + min_distance)
                mood_scores[mood] += similarity * percentage
        
        # Normalize scores
        total = sum(mood_scores.values())
        mood_scores = {k: v/total for k, v in mood_scores.items()}
        
        # Return mood scores and dominant palette
        return {
            'mood_scores': mood_scores,
            'dominant_palette': dominant_colors
        }

    def get_color_hex(self, rgb_color):
        """Convert RGB tuple to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

    def get_mood_summary(self, analysis):
        """Generate a human-readable summary of the mood analysis"""
        mood_scores = analysis['mood_scores']
        dominant_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
        palette_hex = [self.get_color_hex(color) for color, _ in analysis['dominant_palette']]
        
        return {
            'primary_mood': dominant_mood,
            'mood_distribution': mood_scores,
            'color_palette': palette_hex
        }