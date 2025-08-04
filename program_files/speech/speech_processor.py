#!/usr/bin/env python3
"""Speech processing and speaker detection"""

import numpy as np
import webrtcvad
from collections import deque
from typing import Dict, Optional


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
        # WebRTC VAD requires specific frame sizes (10ms, 20ms, or 30ms)
        # For 16kHz, 30ms = 480 samples = 960 bytes
        required_frame_size = 960
        
        try:
            # If frame is too large, take the first valid chunk
            if len(audio_data) >= required_frame_size:
                is_speech = self.vad.is_speech(audio_data[:required_frame_size], self.sample_rate)
            else:
                # If frame is too small, pad with zeros
                padded_data = audio_data + b'\x00' * (required_frame_size - len(audio_data))
                is_speech = self.vad.is_speech(padded_data, self.sample_rate)
        except:
            # Fallback: simple energy-based detection
            import numpy as np
            if len(audio_data) > 0:
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                energy = np.mean(np.abs(audio_np))
                is_speech = energy > 500  # Simple threshold
            else:
                is_speech = False
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.is_speaking = True
        else:
            self.silence_frames += 1
            if self.silence_frames > self.silence_threshold:
                self.is_speaking = False
        
        return is_speech

class SpeakerDetector:
    """Simple speaker detection"""
    
    def __init__(self, sample_rate: int = 16000, enhanced_db=None, speaker_clustering=None):
        self.sample_rate = sample_rate
        self.current_speaker = "Speaker A"
        self.speaker_count = 1
        self.speaker_changes = 0
        self.frames_processed = 0
        self.last_speaker_features = None
        self.feature_buffer = deque(maxlen=10)
        self.enhanced_db = enhanced_db
    
    def extract_voice_features(self, audio_data: bytes) -> Optional[Dict[str, float]]:
        """Extract basic voice characteristics from audio"""
        if len(audio_data) == 0:
            return None
        
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Ensure we don't get NaN values
        energy = float(np.mean(np.abs(audio_np)))
        rms_energy = float(np.sqrt(max(np.mean(audio_np**2), 0)))
        
        return {
            'energy': energy,
            'rms_energy': rms_energy
        }
    
    def update_speaker_count(self, audio_data: bytes, silence_frames: int = 0):
        """Simple speaker tracking"""
        features = self.extract_voice_features(audio_data)
        if features:
            self.feature_buffer.append(features)
            self.frames_processed += 1
        
        # Very basic speaker change detection
        if features and self.last_speaker_features:
            energy_diff = abs(features['energy'] - self.last_speaker_features['energy'])
            if energy_diff > self.last_speaker_features['energy'] * 0.5:
                self.speaker_changes += 1
                self.current_speaker = f"Speaker {chr(65 + (self.speaker_changes % 26))}"
        
        self.last_speaker_features = features
    
    def get_known_speakers(self) -> list:
        """Get list of known speakers"""
        return []
    
    def get_current_features(self) -> Optional[Dict]:
        """Get current features for database storage"""
        if self.feature_buffer:
            return list(self.feature_buffer)[-1]
        return None
    
    def clear_feature_buffer(self):
        """Clear the feature buffer"""
        self.feature_buffer.clear() 