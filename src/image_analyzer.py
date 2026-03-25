import numpy as np
from PIL import Image, ImageEnhance

def analyze_image_quality(image_path):
    """
    Analyze the quality of an image and return a quality score.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        float: The quality score of the image, ranging from 0.0 (low quality) to 1.0 (high quality).
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    gray_image = image.convert('L')
    
    # Calculate the Laplacian of the image to measure sharpness
    laplacian = np.abs(cv2.Laplacian(np.array(gray_image), cv2.CV_64F))
    sharpness_score = np.mean(laplacian)
    
    # Calculate the standard deviation of the pixel values to measure contrast
    contrast_score = np.std(np.array(gray_image))
    
    # Combine the sharpness and contrast scores to get the overall quality score
    quality_score = (sharpness_score + contrast_score) / 2
    
    return quality_score

def enhance_image_quality(image_path, enhancement_factor=1.2):
    """
    Enhance the quality of an image and return the enhanced image.
    
    Args:
        image_path (str): The path to the image file.
        enhancement_factor (float, optional): The factor to use for enhancing the image. Defaults to 1.2.
        
    Returns:
        PIL.Image: The enhanced image.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Enhance the sharpness of the image
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = sharpness_enhancer.enhance(enhancement_factor)
    
    # Enhance the contrast of the image
    contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
    enhanced_image = contrast_enhancer.enhance(enhancement_factor)
    
    return enhanced_image
