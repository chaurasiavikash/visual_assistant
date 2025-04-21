# Visual Assistant

A locally-hosted AI assistant designed to help blind and low vision users by describing images, extracting text, and answering questions about image content.

## Features

- **Image Description**: Generate detailed descriptions of images for blind and low vision users
- **OCR Functionality**: Extract text from images and documents
- **Question Answering**: Ask questions about image content
- **Text-to-Speech**: All results can be read aloud
- **Fully Local**: All processing happens on your device, no external APIs required

## System Requirements

- Python 3.7+
- macOS, Linux, or Windows
- 8GB+ RAM (64GB recommended for optimal performance)
- GPU acceleration (optional, but recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chaurasiavikash/visual_assistant.git
cd visual-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR engine:
```bash
# macOS (using Homebrew)
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

5. Download required models:
```bash
python download_models.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Use the web interface to:
   - Upload images
   - Get descriptions
   - Extract text
   - Ask questions about the image content

## Project Structure

```
visual_assistant/
├── app.py                 # Main FastAPI application
├── download_models.py     # Script to download required models
├── image_processor.py     # Image description module
├── ocr_module.py          # Text extraction module
├── tts_module.py          # Text-to-speech module
├── agent_system.py        # Question answering system
├── utils/                 # Utility functions
│   └── helpers.py
├── static/                # Web interface assets
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/                # Downloaded models (created by download_models.py)
├── uploads/               # Uploaded images (created at runtime)
└── audio_outputs/         # Generated audio files (created at runtime)
```

## Customization

You can customize the application by:

- Modifying the models used in `download_models.py`
- Adjusting preprocessing parameters in `ocr_module.py`
- Changing TTS settings in `tts_module.py`
- Updating the UI in the static files

## Requirements

The main dependencies are:
- FastAPI
- PyTorch
- Transformers
- pytesseract
- pyttsx3
- langchain
- opencv-python

See `requirements.txt` for the complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing pre-trained models
- The PyTorch team for their excellent deep learning framework
- The open-source OCR community for Tesseract