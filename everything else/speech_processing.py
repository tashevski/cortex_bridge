#!/usr/bin/env python3
"""Speech processing utilities for speaker detection and VAD"""

import numpy as np
import webrtcvad
from collections import deque

# Initialize VAD
vad = webrtcvad.Vad(2)

class ConversationManager:
    def __init__(self):
        self.in_gemma_mode = False
        self.gemma_conversation_history = []
        self.exit_keywords = ['exit', 'quit', 'stop', 'bye', 'goodbye', 'end conversation']
        self.enter_keywords = ['hey gemma', 'gemma', 'ai', 'assistant', 'help']
        
    def should_enter_gemma_mode(self, text):
        """Check if we should enter Gemma conversation mode"""
        text_lower = text.lower().strip()
        
        # Check for explicit entry keywords
        if any(keyword in text_lower for keyword in self.enter_keywords):
            return True
        
        # Check if it's a question (existing logic)
        if conditional_pipeline._is_question(text):
            return True
        
        return False
    
    def should_exit_gemma_mode(self, text):
        """Check if we should exit Gemma conversation mode"""
        text_lower = text.lower().strip()
        
        # Check for explicit exit keywords
        if any(keyword in text_lower for keyword in self.exit_keywords):
            return True
        
        # Check for silence or very short responses
        if len(text.strip()) < 3:
            return False  # Don't exit on silence, let user continue
        
        return False
    
    def add_to_history(self, text, is_user=True):
        """Add message to conversation history"""
        role = "user" if is_user else "assistant"
        self.gemma_conversation_history.append({"role": role, "content": text})
        
        # Keep only last 10 messages to avoid context overflow
        if len(self.gemma_conversation_history) > 10:
            self.gemma_conversation_history = self.gemma_conversation_history[-10:]
    
    def get_conversation_context(self):
        """Get conversation context for Gemma"""
        if not self.gemma_conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        for msg in self.gemma_conversation_history[-6:]:  # Last 6 messages
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context

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
        self.speaker_profiles = {}
        self.silence_threshold = 3
        self.feature_change_threshold = 0.2
        self.consecutive_changes = 0
        self.last_change_frame = 0
        self.min_frames_between_changes = 10
        self.speaker_energy_history = deque(maxlen=20)
        self.speaker_pitch_history = deque(maxlen=20)
    
    def extract_voice_features(self, audio_data):
        """Extract enhanced voice characteristics"""
        if len(audio_data) == 0:
            return None
            
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        features = {
            'energy': np.mean(np.abs(audio_np)),
            'pitch_estimate': np.std(audio_np),
            'zero_crossings': np.sum(np.diff(np.sign(audio_np)) != 0),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2])),
            'energy_variance': np.var(np.abs(audio_np)),
            'peak_amplitude': np.max(np.abs(audio_np)),
            'rms_energy': np.sqrt(np.mean(audio_np**2))
        }
        
        return features
    
    def detect_speaker_change_improved(self, current_features):
        """Improved speaker change detection for rapid transitions"""
        if current_features is None:
            return False
        
        self.speaker_energy_history.append(current_features['energy'])
        self.speaker_pitch_history.append(current_features['pitch_estimate'])
        
        # Method 1: Silence-based detection
        if self.silence_frames >= self.silence_threshold:
            if self.last_speaker_features is not None:
                energy_diff = abs(current_features['energy'] - self.last_speaker_features['energy'])
                pitch_diff = abs(current_features['pitch_estimate'] - self.last_speaker_features['pitch_estimate'])
                rms_diff = abs(current_features['rms_energy'] - self.last_speaker_features['rms_energy'])
                
                total_diff = (energy_diff + pitch_diff + rms_diff) / 3
                energy_threshold = self.last_speaker_features['energy'] * self.feature_change_threshold
                
                if total_diff > energy_threshold:
                    return True
        
        # Method 2: Pattern-based detection
        if len(self.speaker_energy_history) >= 10:
            recent_energy = list(self.speaker_energy_history)[-5:]
            older_energy = list(self.speaker_energy_history)[-10:-5]
            
            recent_mean = np.mean(recent_energy)
            older_mean = np.mean(older_energy)
            
            if abs(recent_mean - older_mean) > older_mean * 0.3:
                return True
        
        return False
    
    def update_speaker_count(self, audio_data):
        """Enhanced speaker count estimation"""
        features = self.extract_voice_features(audio_data)
        if features:
            self.voice_characteristics.append(features)
            self.frames_processed += 1
            
            if self.detect_speaker_change_improved(features):
                if self.frames_processed - self.last_change_frame >= self.min_frames_between_changes:
                    self.speaker_changes += 1
                    self.current_speaker = f"Speaker {chr(65 + (self.speaker_changes % 26))}"
                    self.last_change_frame = self.frames_processed
                    
                    self.speaker_profiles[self.current_speaker] = {
                        'features': features.copy(),
                        'first_seen': self.frames_processed,
                        'utterance_count': 0
                    }
                    
                    print(f"🔄 Speaker change detected: {self.current_speaker}")
            
            self.last_speaker_features = features.copy()
            
            if self.frames_processed >= self.min_frames_for_detection:
                unique_voices = self.estimate_unique_voices_improved()
                self.speaker_count = max(1, min(unique_voices, 5))
            else:
                self.speaker_count = 1
    
    def estimate_unique_voices_improved(self):
        """Improved unique voice estimation"""
        if len(self.voice_characteristics) < 5:
            return 1
        
        profile_count = len(self.speaker_profiles)
        if profile_count > 0:
            return profile_count
        
        features_list = list(self.voice_characteristics)
        energy_values = [f['energy'] for f in features_list]
        
        energy_diff = np.diff(sorted(energy_values))
        threshold = np.mean(energy_values) * 0.15
        significant_changes = np.sum(energy_diff > threshold)
        
        return min(significant_changes + 1, 3)
    
    def process_audio_frame(self, audio_data):
        """Process audio frame with improved VAD and speaker detection"""
        try:
            is_speech = vad.is_speech(audio_data, self.sample_rate)
        except:
            is_speech = False
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speaking:
                self.is_speaking = True
        else:
            self.silence_frames += 1
            if self.silence_frames > self.silence_threshold:
                self.is_speaking = False
        
        if is_speech:
            self.update_speaker_count(audio_data)
        
        return is_speech