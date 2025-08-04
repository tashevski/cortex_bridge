#!/usr/bin/env python3
"""Test script for enhanced speaker detection"""

import pyaudio
import json
from vosk import Model, KaldiRecognizer
from speech.speech_processor import SpeechProcessor, SpeakerDetector
import os

def test_enhanced_speaker_detection():
    """Test the enhanced speaker detection system"""
    print("🎤 Enhanced Speaker Detection Test")
    print("=" * 50)
    
    # Initialize components
    speech_processor = SpeechProcessor()
    speaker_detector = SpeakerDetector()
    
    # Check if speaker identification is available
    known_speakers = speaker_detector.get_known_speakers()
    if known_speakers:
        print(f"✅ Advanced speaker identification loaded!")
        print(f"📚 Known speakers: {', '.join(known_speakers)}")
    else:
        print("⚠️  Using basic speaker detection only")
    
    # Load Vosk model
    print("\n🎯 Loading speech recognition model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "vosk-model-small-en-us-0.15")
    model = Model(model_path)
    
    # Initialize speech recognizer
    rec = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()
    
    # Open audio stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=2048)
    stream.start_stream()
    
    print("\n🎤 Speak now! (Ctrl+C to stop)")
    print("The system will identify speakers and show confidence levels.")
    print("-" * 50)
    
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
            is_speech = speech_processor.process_frame(data)
            if is_speech:
                speaker_detector.update_speaker_count(data, speech_processor.silence_frames)
            
            # Process audio through Vosk recognizer
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    text = result['text']
                    
                    # Check for program exit
                    if text.lower() in ["exit", "quit", "stop"]:
                        print("🛑 Stopping test...")
                        break
                    
                    # Display transcription with speaker info
                    print(f"📝 {text}")
                    print(f"   👤 {speaker_detector.current_speaker} | 🎙️ {speaker_detector.speaker_count} voice(s)")
                    
                    # Show speaker profiles if available
                    if speaker_detector.speaker_profiles:
                        print(f"   📊 Active speakers: {list(speaker_detector.speaker_profiles.keys())}")
                    
                    print("-" * 30)
                        
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        
        # Clean shutdown
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("✅ Cleanup complete")
    
    # Final summary
    print("\n📊 Test Summary:")
    print(f"   Total speaker changes: {speaker_detector.speaker_changes}")
    print(f"   Speakers detected: {list(speaker_detector.speaker_profiles.keys())}")
    if known_speakers:
        print(f"   Known speakers: {', '.join(known_speakers)}")

if __name__ == "__main__":
    test_enhanced_speaker_detection() 