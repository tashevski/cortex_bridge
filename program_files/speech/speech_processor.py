#!/usr/bin/env python3
"""Speech processing and speaker detection"""

import numpy as np
import webrtcvad
from collections import deque
from typing import Dict, Any, Optional

class SpeechProcessor:
    """Handles VAD and basic speech processing"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(2)
        
        # VAD state
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.silence_threshold = 3
    
    def process_frame(self, audio_data: bytes) -> bool:
        """Process audio frame and return if speech detected"""
        try:
            is_speech = self.vad.is_speech(audio_data, self.sample_rate)
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
        
        return is_speech

class SpeakerDetector:
    """Enhanced speaker detection with voice characteristics"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.voice_characteristics = deque(maxlen=50)
        self.speaker_count = 1
        self.current_speaker = "Speaker A"
        self.speaker_changes = 0
        
        # Detection parameters
        self.min_frames_for_detection = 5
        self.frames_processed = 0
        self.last_speaker_features = None
        self.speaker_profiles = {}
        self.silence_threshold = 3
        self.feature_change_threshold = 0.2
        self.last_change_frame = 0
        self.min_frames_between_changes = 10
        self.speaker_energy_history = deque(maxlen=20)
        self.speaker_pitch_history = deque(maxlen=20)
    
    def extract_voice_features(self, audio_data: bytes) -> Optional[Dict[str, float]]:
        """Extract voice characteristics from audio"""
        if len(audio_data) == 0:
            return None
            
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        return {
            'energy': np.mean(np.abs(audio_np)),
            'pitch_estimate': np.std(audio_np),
            'zero_crossings': np.sum(np.diff(np.sign(audio_np)) != 0),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2])),
            'energy_variance': np.var(np.abs(audio_np)),
            'peak_amplitude': np.max(np.abs(audio_np)),
            'rms_energy': np.sqrt(np.mean(audio_np**2))
        }
    
    def detect_speaker_change(self, current_features: Dict[str, float]) -> bool:
        """Detect if speaker has changed"""
        if current_features is None:
            return False
        
        self.speaker_energy_history.append(current_features['energy'])
        self.speaker_pitch_history.append(current_features['pitch_estimate'])
        
        # Silence-based detection
        if self.silence_frames >= self.silence_threshold and self.last_speaker_features:
            energy_diff = abs(current_features['energy'] - self.last_speaker_features['energy'])
            pitch_diff = abs(current_features['pitch_estimate'] - self.last_speaker_features['pitch_estimate'])
            rms_diff = abs(current_features['rms_energy'] - self.last_speaker_features['rms_energy'])
            
            total_diff = (energy_diff + pitch_diff + rms_diff) / 3
            energy_threshold = self.last_speaker_features['energy'] * self.feature_change_threshold
            
            if total_diff > energy_threshold:
                return True
        
        # Pattern-based detection
        if len(self.speaker_energy_history) >= 10:
            recent_energy = list(self.speaker_energy_history)[-5:]
            older_energy = list(self.speaker_energy_history)[-10:-5]
            
            recent_mean = np.mean(recent_energy)
            older_mean = np.mean(older_energy)
            
            if abs(recent_mean - older_mean) > older_mean * 0.3:
                return True
        
        return False
    
    def update_speaker_count(self, audio_data: bytes, silence_frames: int = 0):
        """Update speaker detection based on audio"""
        self.silence_frames = silence_frames
        features = self.extract_voice_features(audio_data)
        if not features:
            return
        
        self.voice_characteristics.append(features)
        self.frames_processed += 1
        
        if self.detect_speaker_change(features):
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
            unique_voices = self._estimate_unique_voices()
            self.speaker_count = max(1, min(unique_voices, 5))
        else:
            self.speaker_count = 1
    
    def _estimate_unique_voices(self) -> int:
        """Estimate number of unique voices"""
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