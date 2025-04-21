import cv2
import pytesseract
import numpy as np
import os
from PIL import Image

class OCRProcessor:
    def __init__(self, tesseract_cmd=None):
        """Initialize the OCR processor"""
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # For macOS, tesseract should be installed via Homebrew and found automatically
        
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to get a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert back
        binary = 255 - binary
        
        return binary
    
    def extract_text(self, image_path, preprocess=True, lang='eng'):
        """Extract text from an image"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            if preprocess:
                # Preprocess the image
                processed_image = self.preprocess_image(image)
                # Extract text using pytesseract
                text = pytesseract.image_to_string(processed_image, lang=lang)
            else:
                # Extract text without preprocessing
                text = pytesseract.image_to_string(image, lang=lang)
            
            return text.strip() if text else "No text detected in the image."
        
        except Exception as e:
            return f"Error extracting text: {str(e)}"

# For testing
if __name__ == "__main__":
    ocr = OCRProcessor()
    # Test with a sample image
    test_image = "./data/test.png"  # Replace with an actual image path
    if os.path.exists(test_image):
        text = ocr.extract_text(test_image)
        print(f"Extracted text: {text}")
    else:
        print(f"Test image not found: {test_image}")