import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
import os

class ImageDescriber:
    def __init__(self, clip_model_path="./models/clip", 
                 clip_processor_path="./models/clip_processor",
                 llm_model_path="./models/llm",
                 llm_tokenizer_path="./models/llm_tokenizer"):
        
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor_path)
        
        print("Loading language model...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        
        # Use GPU if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.clip_model.to(self.device)
        self.llm_model.to(self.device)
    
    def get_image_embedding(self, image):
        """Get CLIP embedding for an image"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features
    
    def generate_description(self, image_path):
        """Generate a description for the given image"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Get image features
            image_features = self.get_image_embedding(image)
            
            # Create a prompt for the language model
            prompt = "Describe this image in detail for a blind person: "
            
            # Tokenize the prompt
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                output = self.llm_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                )
            
            description = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clean up by removing the prompt from the output
            description = description.replace(prompt, "").strip()
            
            return description
        
        except Exception as e:
            return f"Error generating description: {str(e)}"

# For testing
if __name__ == "__main__":
    describer = ImageDescriber()
    # Test with a sample image
    test_image = "./data/test.png"  # Replace with an actual image path
    if os.path.exists(test_image):
        description = describer.generate_description(test_image)
        print(f"Description: {description}")
    else:
        print(f"Test image not found: {test_image}")