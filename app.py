import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import shutil
from pydantic import BaseModel
from typing import Optional

# Import our modules
from image_processor import ImageDescriber
from ocr_module import OCRProcessor
from tts_module import TextToSpeech
from agent_system import ImageQuestionAnswerer
from utils.helpers import generate_unique_filename, ensure_directory_exists, is_valid_image

# Create the FastAPI app
app = FastAPI(title="Visual Assistant")

# Create directory for uploaded and processed files
UPLOAD_DIR = ensure_directory_exists("./uploads")
AUDIO_DIR = ensure_directory_exists("./audio_outputs")

# Initialize our components
describer = None
ocr = None
tts = None
qa_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global describer, ocr, tts, qa_system
    
    print("Initializing Visual Assistant components...")
    
    describer = ImageDescriber()
    ocr = OCRProcessor()
    tts = TextToSpeech()
    qa_system = ImageQuestionAnswerer()
    
    print("All components initialized successfully!")

# Mount static files (for the web interface)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Return the main HTML page"""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        # Check if the file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")
        
        # Generate a unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"filename": unique_filename, "path": file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe")
async def describe_image(filename: str = Form(...)):
    """Generate a description for an uploaded image"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not is_valid_image(file_path):
        raise HTTPException(status_code=400, detail="Not a valid image file")
    
    description = describer.generate_description(file_path)
    
    # Generate speech from the description
    audio_filename = f"desc_{os.path.splitext(filename)[0]}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    tts.save_to_file(description, audio_path)
    
    return {
        "description": description,
        "audio_file": audio_filename
    }

@app.post("/extract-text")
async def extract_text(filename: str = Form(...), preprocess: bool = Form(True)):
    """Extract text from an uploaded image"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not is_valid_image(file_path):
        raise HTTPException(status_code=400, detail="Not a valid image file")
    
    text = ocr.extract_text(file_path, preprocess=preprocess)
    
    # Generate speech from the extracted text
    audio_filename = f"text_{os.path.splitext(filename)[0]}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    tts.save_to_file(text, audio_path)
    
    return {
        "text": text,
        "audio_file": audio_filename
    }

@app.post("/answer-question")
async def answer_question(filename: str = Form(...), question: str = Form(...)):
    """Answer a question about an image"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not is_valid_image(file_path):
        raise HTTPException(status_code=400, detail="Not a valid image file")
    
    answer = qa_system.answer_question(file_path, question)
    
    # Generate speech from the answer
    audio_filename = f"answer_{os.path.splitext(filename)[0]}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    tts.save_to_file(answer, audio_path)
    
    return {
        "question": question,
        "answer": answer,
        "audio_file": audio_filename
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_path, media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)