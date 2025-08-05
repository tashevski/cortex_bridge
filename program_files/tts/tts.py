from TTS.api import TTS
import pygame
import os
import re

def clean_text_for_tts(text):
    """Remove emojis and other problematic characters for TTS processing"""
    # Remove emojis and other Unicode symbols
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Remove other problematic characters
    cleaned = emoji_pattern.sub('', text)
    
    # Remove other special characters that might cause issues
    cleaned = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

class OfflineTTSFile:
    def __init__(self):
        try:
            self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            print("TTS initialized successfully!")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.tts = None
    
    def convert_to_file(self, text, filename="output.wav"):
        """Convert text to audio file"""
        if self.tts:
            try:
                # Clean the text before processing
                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    print("⚠️  Text was empty after cleaning, skipping TTS")
                    return False
                
                print(f"Generating speech for: {cleaned_text}")
                self.tts.tts_to_file(text=cleaned_text, file_path=filename)
                print(f"Audio saved to {filename}")
                return True
            except Exception as e:
                print(f"Error generating audio: {e}")
                return False
        return False
    
    def play_file(self, filename):
        """Play audio file using pygame"""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            print("Playback finished!")
        except Exception as e:
            print(f"Error playing file: {e}")

# # Usage
# tts_file = OfflineTTSFile()
# if tts_file.convert_to_file("Hey, How are you going!", "test.wav"):
#     tts_file.play_file("test.wav")
