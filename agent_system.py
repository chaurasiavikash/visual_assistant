import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer

class ImageQuestionAnswerer:
    def __init__(self, clip_model_path="./models/clip", 
                 clip_processor_path="./models/clip_processor",
                 llm_model_path="./models/llm",
                 llm_tokenizer_path="./models/llm_tokenizer"):
        
        print("Loading models for Q&A system...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        
        # Use GPU if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.clip_model.to(self.device)
        self.llm_model.to(self.device)
    
    def get_image_features(self, image_path):
        """Get CLIP features for an image"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            return image_features
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def answer_question(self, image_path, question):
        """Answer a question about an image"""
        try:
            # Get image features
            image_features = self.get_image_features(image_path)
            if image_features is None:
                return "Could not process the image."
            
            # Create a prompt for the language model
            prompt = f"Based on the image, answer this question: {question}\nAnswer: "
            
            # Tokenize the prompt
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                output = self.llm_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                )
            
            answer = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clean up by removing the prompt
            answer = answer.replace(prompt, "").strip()
            
            return answer
        
        except Exception as e:
            return f"Error answering question: {str(e)}"

# For testing
if __name__ == "__main__":
    import os
    
    qa_system = ImageQuestionAnswerer()
    test_image = "./data/test.png"  # Replace with an actual image path
    
    if os.path.exists(test_image):
        # Test with a sample question
        question = "What is the main object in this image?"
        answer = qa_system.answer_question(test_image, question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print(f"Test image not found: {test_image}")