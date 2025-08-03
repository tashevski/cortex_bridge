#!/usr/bin/env python3
"""Program Pipeline with Conditional Gemma Integration - No Emotion Classifier"""

import json
import numpy as np
import pyaudio
import webrtcvad
from vosk import Model, KaldiRecognizer
from collections import deque
from conditional_gemma_input import ConditionalGemmaPipeline, CONDITIONS
import sys
import requests
from conversation_manager import SpeakerDetector, ConversationManager

# Load Vosk model
print("Loading Vosk model...")
model = Model("vosk-model-small-en-us-0.15")

# Initialize VAD
vad = webrtcvad.Vad(2)

# Initialize conditional pipeline
conditional_pipeline = ConditionalGemmaPipeline(
    model="gemma3n:e2b", 
    conditions=CONDITIONS['questions_only']  # Route questions to Gemma
)

def main():
    """Main transcription and conditional Gemma pipeline"""
    print("🎤 Conditional Gemma Pipeline - Speak (Ctrl+C to stop)")
    print("📊 Questions will enter Gemma conversation mode")
    print("🚫 No emotion classifier - clean question detection only")
    print("💬 Say 'exit' to leave Gemma conversation mode")
    
    # Initialize conversation manager
    conversation_manager = ConversationManager()
    
    # Initialize speaker detector
    speaker_detector = SpeakerDetector()
    
    # Initialize speech recognizer
    rec = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()
    
    # Open audio stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=2048)
    stream.start_stream()
    
    try:
        while True:
            # Read audio data
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError as e:
                if e.errno == -9981:
                    continue
                else:
                    print(f"Audio error: {e}")
                    break
            
            # Process VAD and speaker detection
            is_speech = speaker_detector.process_audio_frame(data)
            
            # Process audio through Vosk recognizer
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    text = result['text']
                    
                    # Check for program exit
                    if text.lower() == "exit program":
                        break
                    
                    # Display transcription
                    if conversation_manager.in_gemma_mode:
                        print(f"💬 You: {text}")
                    else:
                        print(f"📝 {text}")
                        print(f"   👤 {speaker_detector.current_speaker} | 🎙️ {speaker_detector.speaker_count} voice(s)")
                    
                    # Conversation state management
                    if conversation_manager.in_gemma_mode:
                        # We're in Gemma conversation mode
                        conversation_manager.add_to_history(text, is_user=True)
                        
                        # Check if we should exit Gemma mode
                        if conversation_manager.should_exit_gemma_mode(text):
                            print("👋 Exiting Gemma conversation mode...")
                            conversation_manager.in_gemma_mode = False
                            conversation_manager.gemma_conversation_history = []
                            print("🎤 Back to listening mode")
                        else:
                            # Continue conversation with Gemma
                            context = conversation_manager.get_conversation_context()
                            prompt = f"{context}\nUser: {text}\n\nAssistant:"
                            
                            try:
                                response = requests.post('http://localhost:11434/api/generate', 
                                    json={'model': 'gemma3n:e4b', 'prompt': prompt, 'stream': False}, timeout=30)
                                
                                if response.status_code == 200:
                                    gemma_response = response.json()['response'].strip()
                                    print(f"🤖 Gemma: {gemma_response}")
                                    conversation_manager.add_to_history(gemma_response, is_user=False)
                                else:
                                    print(f"❌ Error: HTTP {response.status_code}")
                            except Exception as e:
                                print(f"❌ Error: {e}")
                    else:
                        # We're in listening mode - check if we should enter Gemma mode
                        if conversation_manager.should_enter_gemma_mode(text):
                            print("🤖 Entering Gemma conversation mode...")
                            conversation_manager.in_gemma_mode = True
                            conversation_manager.add_to_history(text, is_user=True)
                            
                            # Initial response from Gemma
                            prompt = f"User: {text}\n\nAssistant:"
                            
                            try:
                                response = requests.post('http://localhost:11434/api/generate', 
                                    json={'model': 'gemma3n:e4b', 'prompt': prompt, 'stream': False}, timeout=30)
                                
                                if response.status_code == 200:
                                    gemma_response = response.json()['response'].strip()
                                    print(f"🤖 Gemma: {gemma_response}")
                                    conversation_manager.add_to_history(gemma_response, is_user=False)
                                else:
                                    print(f"❌ Error: HTTP {response.status_code}")
                            except Exception as e:
                                print(f"❌ Error: {e}")
                        else:
                            print("⏭️  Not a question - staying in listening mode")
                        
    except KeyboardInterrupt:
        print("\n🛑 Stopping pipeline...")
        
        # Clean shutdown
        stream.stop_stream()
        stream.close()
        audio.terminate()
        conditional_pipeline.cleanup()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()

