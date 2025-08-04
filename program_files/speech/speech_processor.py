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
        """Extract voice characteristics from audio"""
        if len(audio_data) == 0:
            return None
        
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if len(audio_np) == 0:
            return None
        
        features = {}
        
        # Basic energy features
        features['energy'] = float(np.mean(np.abs(audio_np)))
        features['rms_energy'] = float(np.sqrt(max(np.mean(audio_np**2), 0)))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np))))
        features['zcr'] = float(zero_crossings / max(len(audio_np) - 1, 1))
        
        # Spectral features (requires FFT)
        try:
            fft = np.fft.rfft(audio_np)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_np), 1/16000)
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                features['spectral_centroid'] = float(np.sum(freqs * magnitude) / np.sum(magnitude))
            else:
                features['spectral_centroid'] = 0.0
            
            # Spectral rolloff (95% of energy)
            cumsum = np.cumsum(magnitude**2)
            if cumsum[-1] > 0:
                rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                features['spectral_rolloff'] = float(freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
            else:
                features['spectral_rolloff'] = 0.0
        except:
            features['spectral_centroid'] = 0.0
            features['spectral_rolloff'] = 0.0
        
        # Simple pitch estimation (autocorrelation peak)
        try:
            autocorr = np.correlate(audio_np, audio_np, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak in typical pitch range (80-400 Hz)
            min_lag = int(16000 / 400)  # 400 Hz
            max_lag = int(16000 / 80)   # 80 Hz
            if max_lag < len(autocorr):
                pitch_autocorr = autocorr[min_lag:max_lag]
                if len(pitch_autocorr) > 0:
                    peak_idx = np.argmax(pitch_autocorr) + min_lag
                    features['pitch'] = float(16000 / peak_idx) if peak_idx > 0 else 0.0
                else:
                    features['pitch'] = 0.0
            else:
                features['pitch'] = 0.0
        except:
            features['pitch'] = 0.0
        
        # MFCC (simplified - first 3 coefficients)
        try:
            # Mel filterbank (simplified)
            n_mels = 13
            fft_freqs = np.fft.rfftfreq(len(audio_np), 1/16000)
            mel_freqs = np.linspace(0, 2595 * np.log10(1 + 8000/700), n_mels)
            hz_freqs = 700 * (10**(mel_freqs/2595) - 1)
            
            mel_filters = np.zeros((n_mels-2, len(fft_freqs)))
            for m in range(1, n_mels-1):
                f_left = hz_freqs[m-1]
                f_center = hz_freqs[m]
                f_right = hz_freqs[m+1]
                
                for k, freq in enumerate(fft_freqs):
                    if f_left <= freq < f_center:
                        mel_filters[m-1, k] = (freq - f_left) / (f_center - f_left)
                    elif f_center <= freq <= f_right:
                        mel_filters[m-1, k] = (f_right - freq) / (f_right - f_center)
            
            # Apply mel filters
            magnitude = np.abs(np.fft.rfft(audio_np))
            mel_energies = np.dot(mel_filters, magnitude)
            log_mel = np.log(np.maximum(mel_energies, 1e-10))
            
            # DCT for MFCCs (first 3)
            mfccs = np.zeros(3)
            for i in range(3):
                for j in range(len(log_mel)):
                    mfccs[i] += log_mel[j] * np.cos(i * (j + 0.5) * np.pi / len(log_mel))
            
            features['mfcc1'] = float(mfccs[0])
            features['mfcc2'] = float(mfccs[1]) 
            features['mfcc3'] = float(mfccs[2])
        except:
            features['mfcc1'] = 0.0
            features['mfcc2'] = 0.0
            features['mfcc3'] = 0.0
        
        # Clean any NaN/inf values
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
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