from TTS.api import TTS
import pygame
import os
import re
import threading
import time
from queue import Queue

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

def split_text_into_chunks(text, max_chunk_length=100):
    """Split text into smaller chunks for streaming TTS"""
    # First, protect common abbreviations and phrases from being split
    protected_phrases = [
        "e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Ph.D.",
        "a.m.", "p.m.", "U.S.", "U.K.", "U.S.A.", "U.K.", "etc.", "vs.", "vs",
        "e.g", "i.e", "a.m", "p.m", "U.S", "U.K", "U.S.A"
    ]
    
    # Temporarily replace protected phrases with placeholders
    protected_map = {}
    for i, phrase in enumerate(protected_phrases):
        placeholder = f"__PROTECTED_{i}__"
        text = text.replace(phrase, placeholder)
        protected_map[placeholder] = phrase
    
    # Split by sentences, but be more careful about sentence boundaries
    # Look for sentence endings followed by space and capital letter
    import re
    
    # Split on sentence endings (.!?) followed by space and capital letter
    sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)
    
    # Reconstruct sentences properly
    reconstructed_sentences = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            # Combine sentence with its punctuation
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        if sentence.strip():
            reconstructed_sentences.append(sentence.strip())
    
    # Restore protected phrases
    for placeholder, phrase in protected_map.items():
        for i, sentence in enumerate(reconstructed_sentences):
            reconstructed_sentences[i] = sentence.replace(placeholder, phrase)
    
    chunks = []
    current_chunk = ""
    
    for sentence in reconstructed_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Skip very short sentences that are likely fragments
        if len(sentence) < 3:
            continue
            
        # If adding this sentence would exceed the limit, save current chunk and start new one
        if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists and is substantial
    if current_chunk and len(current_chunk.strip()) >= 3:
        chunks.append(current_chunk.strip())
    
    # Filter out chunks that are too short or just punctuation
    filtered_chunks = []
    for chunk in chunks:
        # Remove extra whitespace and check if it's substantial
        cleaned_chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(cleaned_chunk) >= 5:  # Minimum 5 characters
            filtered_chunks.append(cleaned_chunk)
    
    return filtered_chunks

class OfflineTTSFile:
    def __init__(self):
        try:
            self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
            print("TTS initialized successfully!")
            self.tts_available = True
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            print("⚠️  TTS will be disabled. Speech responses will be text-only.")
            self.tts = None
            self.tts_available = False
        
        # Initialize pygame mixer for streaming
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
        except Exception as e:
            print(f"Error initializing pygame mixer: {e}")
            self.mixer_initialized = False
    
    def convert_to_file(self, text, filename="output.wav", speaker="p225"):
        """Convert text to audio file"""
        if self.tts and self.tts_available:
            try:
                # Clean the text before processing
                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    print("⚠️  Text was empty after cleaning, skipping TTS")
                    return False
                
                print(f"Generating speech for: {cleaned_text}")
                self.tts.tts_to_file(text=cleaned_text, file_path=filename, speaker=speaker)
                print(f"Audio saved to {filename}")
                return True
            except Exception as e:
                print(f"Error generating audio: {e}")
                return False
        return False
    
    def play_file(self, filename):
        """Play audio file using pygame"""
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            print("Playback finished!")
        except Exception as e:
            print(f"Error playing file: {e}")
    
    def stream_text_to_speech(self, text, chunk_length=100, speaker="p229"):
        """Stream text to speech in chunks for real-time playback"""
        if not self.tts_available or not self.mixer_initialized:
            print("❌ TTS or mixer not initialized")
            return False
        
        try:
            # Clean the text
            cleaned_text = clean_text_for_tts(text)
            if not cleaned_text:
                print("⚠️  Text was empty after cleaning, skipping TTS")
                return False
            
            # Split into chunks
            chunks = split_text_into_chunks(cleaned_text, chunk_length)
            
            if not chunks:
                return False
            
            print(f"🔊 Streaming {len(chunks)} chunks...")
            
            # Process and play each chunk
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Generate unique filename for this chunk
                chunk_filename = f"chunk_{int(time.time())}_{i}.wav"
                
                try:
                    print(f"🎵 Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                    
                    # Generate audio for this chunk with speaker parameter
                    self.tts.tts_to_file(text=chunk, file_path=chunk_filename, speaker=speaker)
                    
                    # Play the chunk
                    pygame.mixer.music.load(chunk_filename)
                    pygame.mixer.music.play()
                    
                    # Wait for this chunk to finish playing
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    # Clean up the chunk file
                    try:
                        os.remove(chunk_filename)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"❌ Error processing chunk {i+1}: {e}")
                    # Clean up file if it exists
                    try:
                        os.remove(chunk_filename)
                    except:
                        pass
                    continue
            
            print("✅ Streaming TTS complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error in streaming TTS: {e}")
            return False

# # Usage
# tts_file = OfflineTTSFile()
# if tts_file.convert_to_file("Hey, How are you going!", "test.wav"):
#     tts_file.play_file("test.wav")
