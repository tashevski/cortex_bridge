#!/usr/bin/env python3
"""Improved transcriber with enhanced speaker detection for rapid changes"""

import json
import numpy as np
import pyaudio
import webrtcvad
from vosk import Model, KaldiRecognizer
from transformers import pipeline
from collections import deque
from conversation_logger import log_conversation, start_conversation_session, end_conversation_session

# Load Vosk model
print("Loading Vosk model...")
model = Model("vosk-model-small-en-us-0.15")

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
        self.voice_characteristics = deque(maxlen=50)
        self.speaker_count = 1
        self.current_speaker = "Speaker A"
        self.speaker_changes = 0
        
        # VAD state
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Minimum frames needed for voice detection
        self.min_frames_for_detection = 5
        self.frames_processed = 0
        
        # IMPROVED: Better speaker detection for rapid changes
        self.last_speaker_features = None
        self.speaker_profiles = {}  # Store profiles for each detected speaker
        self.silence_threshold = 3  # Reduced from 10 to 3 frames (90ms) for faster detection
        self.feature_change_threshold = 0.2  # Lower threshold for faster detection
        self.consecutive_changes = 0
        self.last_change_frame = 0
        self.min_frames_between_changes = 10  # Prevent too frequent changes
        self.speaker_energy_history = deque(maxlen=20)  # Track energy patterns
        self.speaker_pitch_history = deque(maxlen=20)   # Track pitch patterns
    
    def extract_voice_features(self, audio_data):
        """Extract enhanced voice characteristics"""
        if len(audio_data) == 0:
            return None
            
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Enhanced features for better speaker discrimination
        features = {
            'energy': np.mean(np.abs(audio_np)),
            'pitch_estimate': np.std(audio_np),
            'zero_crossings': np.sum(np.diff(np.sign(audio_np)) != 0),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2])),
            'energy_variance': np.var(np.abs(audio_np)),  # NEW: Energy variation
            'peak_amplitude': np.max(np.abs(audio_np)),   # NEW: Peak amplitude
            'rms_energy': np.sqrt(np.mean(audio_np**2))   # NEW: RMS energy
        }
        
        return features
    
    def detect_speaker_change_improved(self, current_features):
        """Improved speaker change detection for rapid transitions"""
        if current_features is None:
            return False
        
        # Store energy and pitch for pattern analysis
        self.speaker_energy_history.append(current_features['energy'])
        self.speaker_pitch_history.append(current_features['pitch_estimate'])
        
        # Method 1: Silence-based detection (most reliable for rapid changes)
        if self.silence_frames >= self.silence_threshold:
            # After silence, assume potential speaker change
            if self.last_speaker_features is not None:
                # Compare with last speaker's features
                energy_diff = abs(current_features['energy'] - self.last_speaker_features['energy'])
                pitch_diff = abs(current_features['pitch_estimate'] - self.last_speaker_features['pitch_estimate'])
                rms_diff = abs(current_features['rms_energy'] - self.last_speaker_features['rms_energy'])
                
                # Combined difference score
                total_diff = (energy_diff + pitch_diff + rms_diff) / 3
                energy_threshold = self.last_speaker_features['energy'] * self.feature_change_threshold
                
                if total_diff > energy_threshold:
                    return True
        
        # Method 2: Pattern-based detection (for overlapping speech)
        if len(self.speaker_energy_history) >= 10:
            # Look for energy pattern changes
            recent_energy = list(self.speaker_energy_history)[-5:]
            older_energy = list(self.speaker_energy_history)[-10:-5]
            
            recent_mean = np.mean(recent_energy)
            older_mean = np.mean(older_energy)
            
            if abs(recent_mean - older_mean) > older_mean * 0.3:
                return True
        
        # Method 3: Traditional feature comparison (fallback)
        if len(self.voice_characteristics) >= 3:
            recent_features = list(self.voice_characteristics)[-3:]
            differences = []
            
            for old_features in recent_features:
                diff = sum(abs(current_features[k] - old_features[k]) for k in current_features.keys())
                differences.append(diff)
            
            avg_diff = np.mean(differences)
            threshold = np.mean([f['energy'] for f in recent_features]) * self.feature_change_threshold
            
            if avg_diff > threshold:
                return True
        
        return False
    
    def update_speaker_count(self, audio_data):
        """Enhanced speaker count estimation"""
        features = self.extract_voice_features(audio_data)
        if features:
            self.voice_characteristics.append(features)
            self.frames_processed += 1
            
            # IMPROVED: Better speaker change detection
            if self.detect_speaker_change_improved(features):
                # Prevent too frequent changes
                if self.frames_processed - self.last_change_frame >= self.min_frames_between_changes:
                    self.speaker_changes += 1
                    self.current_speaker = f"Speaker {chr(65 + (self.speaker_changes % 26))}"
                    self.last_change_frame = self.frames_processed
                    
                    # Store speaker profile
                    self.speaker_profiles[self.current_speaker] = {
                        'features': features.copy(),
                        'first_seen': self.frames_processed,
                        'utterance_count': 0
                    }
                    
                    print(f"🔄 Speaker change detected: {self.current_speaker}")
            
            # Update last speaker features
            self.last_speaker_features = features.copy()
            
            # Estimate speaker count
            if self.frames_processed >= self.min_frames_for_detection:
                unique_voices = self.estimate_unique_voices_improved()
                self.speaker_count = max(1, min(unique_voices, 5))
            else:
                self.speaker_count = 1
    
    def estimate_unique_voices_improved(self):
        """Improved unique voice estimation"""
        if len(self.voice_characteristics) < 5:
            return 1
        
        # Count distinct speaker profiles
        profile_count = len(self.speaker_profiles)
        if profile_count > 0:
            return profile_count
        
        # Fallback to energy-based estimation
        features_list = list(self.voice_characteristics)
        energy_values = [f['energy'] for f in features_list]
        
        # Use clustering to identify distinct energy levels
        energy_diff = np.diff(sorted(energy_values))
        threshold = np.mean(energy_values) * 0.15  # Lower threshold
        significant_changes = np.sum(energy_diff > threshold)
        
        return min(significant_changes + 1, 3)
    
    def process_audio_frame(self, audio_data):
        """Process audio frame with improved VAD and speaker detection"""
        # VAD detection
        try:
            is_speech = vad.is_speech(audio_data, self.sample_rate)
        except:
            is_speech = False
        
        # Update VAD state with improved silence detection
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speaking:
                self.is_speaking = True
        else:
            self.silence_frames += 1
            # IMPROVED: Faster silence detection for rapid speaker changes
            if self.silence_frames > self.silence_threshold:
                self.is_speaking = False
        
        # Update speaker detection
        if is_speech:
            self.update_speaker_count(audio_data)
        
        return is_speech

# Initialize speaker detector
speaker_detector = SpeakerDetector()

def analyze_text(text):
    """Analyze text for emotion and question detection using ML model"""
    # Get emotion classification from ML model
    emotions = emotion_classifier(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])
    
    # Emotion emoji mapping
    emotion_text = f"{top_emotion['label'].title()}"
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

# Initialize speech recognizer with 16kHz sample rate
rec = KaldiRecognizer(model, 16000)
audio = pyaudio.PyAudio()

# Open audio stream from microphone with larger buffer for stability
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                   input=True, frames_per_buffer=2048)
stream.start_stream()

# Start conversation logging session
session_id = start_conversation_session()

print("🎤 Speak (OFFLINE)... (Ctrl+C to stop)")
print("📊 ML-based emotion classification + VAD enabled")
print(f"📝 Logging to session: {session_id}")

try:
    while True:
        # Read audio data in chunks with overflow handling
        try:
            data = stream.read(2048, exception_on_overflow=False)
        except OSError as e:
            if e.errno == -9981:  # Input overflow
                print("⚠️  Audio buffer overflow - continuing...")
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
                emotion_text, confidence, question_mark = analyze_text(text)
                
                # Get speaker info
                speaker_info = f"👤 {speaker_detector.current_speaker}"
                voice_count = f"🎙️ {speaker_detector.speaker_count} voice(s)"
                
                # Log conversation data
                log_conversation(
                    text=text,
                    speaker=speaker_detector.current_speaker,
                    emotion=emotion_text,
                    emotion_confidence=confidence,
                    is_question=bool(question_mark),
                    voice_count=speaker_detector.speaker_count
                )
                
                # Display transcription with analysis
                print(f"📝 {text}")
                print(f"   {speaker_info} | {voice_count} | {emotion_text} ({confidence:.2f}) {question_mark}")
                print()
                
except KeyboardInterrupt:
    print("\n🛑 Stopping transcriber...")
    # End conversation session
    end_conversation_session()
    
    # Clean shutdown
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("✅ Cleanup complete") 