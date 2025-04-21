# Create a script to download models (download_models.py)
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
import torch

# Download CLIP
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Save locally
clip_model.save_pretrained("./models/clip")
clip_processor.save_pretrained("./models/clip_processor")

# Download a text generation model 
# For image captioning when combined with CLIP
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for local inference
llm_model = AutoModelForCausalLM.from_pretrained(model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
llm_model.save_pretrained("./models/llm")
llm_tokenizer.save_pretrained("./models/llm_tokenizer")

print("Models downloaded successfully!")