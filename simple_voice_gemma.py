#!/usr/bin/env python3
"""Minimal voice chat with Gemma and ML-based emotion classification + VAD"""

import pyaudio
import speech_recognition as sr
import numpy as np
import threading
import queue
import requests
import subprocess
import time
from transformers import pipeline
import webrtcvad
from collections import deque
from conversation_logger import log_conversation, start_conversation_session, end_conversation_session

# Load emotion classification model
print("Loading emotion classification model...")
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (moderate)

class SpeakerDetector:
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Voice characteristics tracking
        self.voice_characteristics = deque(maxlen=50)  # Store recent voice features
        self.speaker_count = 0
        self.current_speaker = "Speaker A"
        self.speaker_changes = 0
        
        # VAD state
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
    def extract_voice_features(self, audio_data):
        """Extract basic voice characteristics"""
        if len(audio_data) == 0:
            return None
            
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Basic features
        features = {
            'energy': np.mean(np.abs(audio_np)),
            'pitch_estimate': np.std(audio_np),  # Simplified pitch estimate
            'zero_crossings': np.sum(np.diff(np.sign(audio_np)) != 0),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2]))
        }
        
        return features
    
    def detect_speaker_change(self, current_features):
        """Detect if speaker has changed based on voice characteristics"""
        if current_features is None or len(self.voice_characteristics) < 5:
            return False
            
        # Compare with recent voice characteristics
        recent_features = list(self.voice_characteristics)[-5:]
        
        # Calculate differences
        differences = []
        for old_features in recent_features:
            diff = sum(abs(current_features[k] - old_features[k]) for k in current_features.keys())
            differences.append(diff)
        
        # If significant change detected
        avg_diff = np.mean(differences)
        threshold = np.mean([f['energy'] for f in recent_features]) * 0.3
        
        return avg_diff > threshold
    
    def update_speaker_count(self, audio_data):
        """Estimate number of speakers based on voice variety"""
        features = self.extract_voice_features(audio_data)
        if features:
            self.voice_characteristics.append(features)
            
            # Detect speaker change
            if self.detect_speaker_change(features):
                self.speaker_changes += 1
                self.current_speaker = f"Speaker {chr(65 + (self.speaker_changes % 26))}"
            
            # Estimate speaker count based on voice variety
            if len(self.voice_characteristics) > 10:
                unique_voices = self.estimate_unique_voices()
                self.speaker_count = max(1, min(unique_voices, 5))  # Cap at 5 speakers
    
    def estimate_unique_voices(self):
        """Estimate number of unique voices based on feature clustering"""
        if len(self.voice_characteristics) < 5:
            return 1
            
        features_list = list(self.voice_characteristics)
        
        # Simple clustering based on energy differences
        energy_values = [f['energy'] for f in features_list]
        energy_diff = np.diff(sorted(energy_values))
        
        # Count significant energy level changes
        threshold = np.mean(energy_values) * 0.2
        significant_changes = np.sum(energy_diff > threshold)
        
        return min(significant_changes + 1, 3)  # Estimate 1-3 speakers
    
    def process_audio_frame(self, audio_data):
        """Process audio frame for VAD and speaker detection"""
        # VAD detection
        try:
            is_speech = vad.is_speech(audio_data, self.sample_rate)
        except:
            is_speech = False
        
        # Update VAD state
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speaking:
                self.is_speaking = True
        else:
            self.silence_frames += 1
            if self.silence_frames > 10:  # 300ms of silence
                self.is_speaking = False
        
        # Update speaker detection
        if is_speech:
            self.update_speaker_count(audio_data)
        
        return is_speech

def analyze_text(text):
    """Analyze text for emotion and question detection using ML model"""
    # Get emotion classification from ML model
    emotions = emotion_classifier(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])
    
    # Emotion emoji mapping
    emotion_emojis = {
        'joy': '😊',
        'sadness': '😢', 
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐'
    }
    
    emotion_emoji = emotion_emojis.get(top_emotion['label'], '😐')
    emotion_text = f"{emotion_emoji} {top_emotion['label'].title()}"
    confidence = top_emotion['score']
    
    # Question detection
    text_lower = text.lower()
    is_question = (
        text.strip().endswith('?') or
        text_lower.startswith(('what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose')) or
        text_lower.startswith(('is', 'are', 'can', 'could', 'would', 'should', 'do', 'does', 'did'))
    )
    
    # Question indicator
    question_mark = "❓ Question" if is_question else ""
    
    return emotion_text, confidence, question_mark

class SimpleVoiceGemma:
    def __init__(self, model="gemma3n:e2b"):
        self.model = model
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Initialize speaker detector
        self.speaker_detector = SpeakerDetector()
        
        # Initialize conversation logging
        self.session_id = None
        
    def start(self):
        # Start Ollama server and pull model
        subprocess.Popen(['ollama', 'serve'])
        time.sleep(2)
        subprocess.run(['ollama', 'pull', self.model])
        
        self.running = True
        # Open audio stream for microphone input
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=16000,
            input=True, frames_per_buffer=1024,
            stream_callback=self._callback
        )
        
        # Start processing thread
        threading.Thread(target=self._process, daemon=True).start()
        self.stream.start_stream()
        
        # Start conversation logging session
        self.session_id = start_conversation_session()
        
        print("🎤 Speak to Gemma... (Ctrl+C to stop)")
        print("📊 ML-based emotion classification + VAD enabled")
        print(f"📝 Logging to session: {self.session_id}")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def _callback(self, in_data, frame_count, time_info, status):
        # Audio callback - detect when there's sound
        if self.running:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Process VAD and speaker detection
            is_speech = self.speaker_detector.process_audio_frame(in_data)
            
            if np.abs(audio_data).mean() > 1000:  # Noise threshold
                self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process(self):
        buffer = []
        while self.running:
            try:
                data = self.audio_queue.get(timeout=1)
                buffer.extend(data)
                
                # Process 2 seconds of audio at a time
                if len(buffer) > 16000 * 2:
                    try:
                        # Convert audio to text using Google Speech Recognition
                        audio = sr.AudioData(
                            np.array(buffer, dtype=np.int16).tobytes(),
                            16000, 2
                        )
                        text = self.recognizer.recognize_google(audio)
                        if text:
                            # Analyze emotion and question
                            emotion_text, confidence, question_mark = analyze_text(text)
                            
                            # Get speaker info
                            speaker_info = f"👤 {self.speaker_detector.current_speaker}"
                            voice_count = f"🎙️ {self.speaker_detector.speaker_count} voice(s)"
                            
                            print(f"🎤 You: {text}")
                            print(f"   {speaker_info} | {voice_count} | {emotion_text} ({confidence:.2f}) {question_mark}")
                            
                            # Log conversation data
                            log_conversation(
                                text=text,
                                speaker=self.speaker_detector.current_speaker,
                                emotion=emotion_text,
                                emotion_confidence=confidence,
                                is_question=bool(question_mark),
                                voice_count=self.speaker_detector.speaker_count
                            )
                            
                            # Send text to Gemma and get response
                            response = requests.post(
                                'http://localhost:11434/api/generate',
                                json={'model': self.model, 'prompt': text, 'stream': False}
                            ).json()
                            
                            print(f"🤖 Gemma: {response['response']}\n")
                    except:
                        pass
                    buffer = []
            except queue.Empty:
                pass
    
    def stop(self):
        # End conversation session
        if self.session_id:
            end_conversation_session()
        
        # Clean shutdown
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    SimpleVoiceGemma().start() 