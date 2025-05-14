"""
OCR implementation for handwritten answers in the Theoretical Answer Evaluation System.
"""

import pytesseract
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def extract_text_from_image(image):
    """
    Extract text from an image using OCR.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        str: Extracted text
    """
    # Apply preprocessing to enhance text readability
    processed_image = preprocess_image_for_ocr(image)
    
    # Extract text using pytesseract
    try:
        # Set configuration for pytesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def preprocess_image_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image (PIL.Image): Original image
    
    Returns:
        PIL.Image: Processed image
    """
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize for better OCR performance if image is too small
    width, height = image.size
    if width < 1000:
        ratio = 1000 / width
        new_width = 1000
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Apply noise removal
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Apply thresholding to make text more distinct
    threshold = 150
    image = image.point(lambda p: 0 if p < threshold else 255)
    
    # Apply dilation and erosion to connect broken text parts
    image = image.filter(ImageFilter.MaxFilter(size=3))
    image = image.filter(ImageFilter.MinFilter(size=2))
    
    return image

def improve_ocr_result(text):
    """
    Post-process OCR results to correct common errors.
    
    Args:
        text (str): Raw OCR text
    
    Returns:
        str: Improved text
    """
    # Replace common OCR errors
    replacements = {
        '0': 'o',  # Zero to o
        '1': 'l',  # One to l
        '@': 'a',  # @ to a
        '$': 's',  # $ to s
        '|': 'l',  # | to l
        '{': '(',  # { to (
        '}': ')',  # } to )
        '[': '(',  # [ to (
        ']': ')',  # ] to )
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix line breaks
    text = text.replace('-\n', '')
    
    # Fix multiple spaces
    text = ' '.join(text.split())
    
    return text

def extract_text_with_confidence(image):
    """
    Extract text with confidence scores.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        tuple: (text, confidence)
    """
    processed_image = preprocess_image_for_ocr(image)
    
    try:
        # Get confidence data
        data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Skip entries with -1 confidence
                text_parts.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return full_text, avg_confidence
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", 0

def is_handwritten(image):
    """
    Attempt to determine if the image contains handwritten or printed text.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        bool: True if likely handwritten, False if likely printed
    """
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Get image array
    img_array = np.array(gray_image)
    
    # Calculate standard deviation of pixel values
    std_dev = np.std(img_array)
    
    # Calculate histogram of pixel values
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    
    # Normalize histogram
    hist = hist / hist.sum()
    
    # Calculate entropy (measure of disorder)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Handwritten text typically has higher entropy and std deviation
    if entropy > 7.0 and std_dev > 50:
        return True
    else:
        return False

if __name__ == "__main__":
    # Example usage
    try:
        # Load a test image
        test_image = Image.open("test_handwriting.jpg")
        
        # Extract text
        text = extract_text_from_image(test_image)
        print("Extracted text:")
        print(text)
        
        # Extract with confidence
        text, confidence = extract_text_with_confidence(test_image)
        print(f"Confidence: {confidence:.2f}%")
        
        # Check if handwritten
        handwritten = is_handwritten(test_image)
        print(f"Handwritten: {handwritten}")
    except Exception as e:
        print(f"Error in test: {e}")
