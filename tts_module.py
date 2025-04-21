import pyttsx3
import os
import time

class TextToSpeech:
    def __init__(self, voice=None, rate=150):
        """Initialize the TTS engine"""
        self.engine = pyttsx3.init()
        
        # Set rate
        self.engine.setProperty('rate', rate)
        
        # Set voice if specified
        if voice:
            self.engine.setProperty('voice', voice)
    
    def list_available_voices(self):
        """List all available voices"""
        voices = self.engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"Voice {i}:")
            print(f" - ID: {voice.id}")
            print(f" - Name: {voice.name}")
            print(f" - Languages: {voice.languages}")
            print(f" - Gender: {voice.gender}")
            print(f" - Age: {voice.age}")
            print("-------------------------")
        return voices
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        if not text:
            return False
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            return False
    
    def save_to_file(self, text, output_file="output.mp3"):
        """Save speech to an audio file"""
        try:
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return os.path.abspath(output_file)
        except Exception as e:
            print(f"Error saving speech to file: {str(e)}")
            return None

# For testing
if __name__ == "__main__":
    tts = TextToSpeech()
    
    # List available voices
    print("Available voices:")
    tts.list_available_voices()
    
    # Test speaking
    test_text = "Hello! This is a test of the text to speech system."
    tts.speak_text(test_text)
    
    # Test saving to file
    file_path = tts.save_to_file("This text is saved to an audio file.")
    if file_path:
        print(f"Audio saved to: {file_path}")