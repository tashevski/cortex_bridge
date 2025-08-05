#!/usr/bin/env python3
"""Minimal speech processing and speaker detection"""

import numpy as np
import webrtcvad
from typing import Dict, Optional
from config.config import SpeechProcessorConfig, SpeakerDetectorConfig


class SpeechProcessor:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, config: Optional[SpeechProcessorConfig] = None):
        if config is None:
            from config.config import cfg
            config = cfg.speech_processor
            
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.is_speaking = False
        self.silence_frames = 0
    
    def process_frame(self, audio_data: bytes) -> bool:
        """Process audio frame and return if speech detected"""
        try:
            frame_size = self.config.frame_size
            frame_data = audio_data[:frame_size] if len(audio_data) >= frame_size else audio_data.ljust(frame_size, b'\x00')
            is_speech = self.vad.is_speech(frame_data, self.config.sample_rate)
        except:
            audio_np = np.frombuffer(audio_data, dtype=np.int16) if audio_data else np.array([])
            is_speech = np.mean(np.abs(audio_np)) > self.config.energy_threshold if len(audio_np) > 0 else False
        
        if is_speech:
            self.silence_frames = 0
            self.is_speaking = True
        else:
            self.silence_frames += 1
            if self.silence_frames > self.config.silence_threshold:
                self.is_speaking = False
        
        return is_speech


class SpeakerDetector:
    """Real-time speaker detection using ECAPA-TDNN embeddings"""
    
    def __init__(self, config: Optional[SpeakerDetectorConfig] = None, **kwargs):
        if config is None:
            from config.config import cfg
            config = cfg.speaker_detector
            
        self.config = config
        
        # Core settings
        self.current_speaker = "Speaker_A"
        self.speaker_count = 1
        self.speaker_profiles = []
        self.audio_buffer = []
        
        # State tracking
        self.frames_since_change = 0
        self.new_speaker_candidates = {}
        self.speaker_changed = False
        
        # Load ECAPA-TDNN model if enabled
        self.speaker_model = None
        if config.use_ecapa_model:
            try:
                import speechbrain.pretrained
                self.speaker_model = speechbrain.pretrained.EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=config.model_save_dir
                )
            except:
                print("⚠️  ECAPA-TDNN model not available, falling back to spectral features")
                self.speaker_model = None
    
    def _get_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio"""
        if self.speaker_model:
            # Use ECAPA-TDNN
            import torch
            audio_norm = audio_np / (np.max(np.abs(audio_np)) + 1e-10)
            audio_tensor = torch.tensor(audio_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor).squeeze().cpu().numpy()
                if self.config.normalize_embeddings:
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)  # L2 normalize
                return embedding
        else:
            # Fallback: spectral features
            fft = np.fft.rfft(audio_np, n=self.config.fft_size)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(self.config.fft_size, 1/16000)
            
            # Focus on voice range
            voice_mask = (freqs >= self.config.voice_freq_min) & (freqs <= self.config.voice_freq_max)
            voice_mag = magnitude[voice_mask]
            voice_freqs = freqs[voice_mask]
            
            # Extract frequency band energies
            bands = np.array_split(voice_mag, self.config.spectral_bands)
            band_energies = [np.mean(band) for band in bands]
            
            # Add spectral features
            if np.sum(voice_mag) > 0:
                centroid = np.sum(voice_freqs * voice_mag) / np.sum(voice_mag)
                spread = np.sqrt(np.sum(((voice_freqs - centroid) ** 2) * voice_mag) / np.sum(voice_mag))
            else:
                centroid = spread = 0
                
            return np.array(band_energies + [centroid, spread])
    
    def identify_speaker(self, audio_data: bytes) -> str:
        """Identify speaker from audio frame"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) == 0:
            return self.current_speaker
        
        self.audio_buffer.extend(audio_np)
        if len(self.audio_buffer) > self.config.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.config.buffer_size:]
        
        if len(self.audio_buffer) < self.config.buffer_size:
            return self.current_speaker
        
        buffer_array = np.array(self.audio_buffer)
        energy = np.sqrt(np.mean(buffer_array ** 2))
        if energy < self.config.min_speech_energy or np.max(np.abs(buffer_array)) < 0.05:
            self.new_speaker_candidates.clear()
            return self.current_speaker
        
        # Get speaker embedding
        current_embedding = self._get_embedding(buffer_array)
        
        # First speaker (with energy check)
        if not self.speaker_profiles:
            if energy >= self.config.min_speech_energy * 2:
                self.speaker_profiles.append({'id': 'Speaker_A', 'embedding': current_embedding, 'count': 1})
                self.speaker_count = 1
                return 'Speaker_A'
            return self.current_speaker
        
        # Compare with existing speakers
        similarities = [np.dot(current_embedding, profile['embedding']) for profile in self.speaker_profiles]
        max_similarity = max(similarities)
        
        if max_similarity >= self.config.similarity_threshold:
            # Match existing speaker
            best_idx = similarities.index(max_similarity)
            profile = self.speaker_profiles[best_idx]
            
            # Update embedding with moving average
            alpha = self.config.embedding_alpha
            profile['embedding'] = (1 - alpha) * profile['embedding'] + alpha * current_embedding
            profile['count'] += 1
            
            candidate_speaker = profile['id']
            self.new_speaker_candidates.clear()
        else:
            # Potential new speaker
            if len(self.speaker_profiles) < self.config.max_speakers:
                # Track candidate
                if 'potential_new' not in self.new_speaker_candidates:
                    self.new_speaker_candidates['potential_new'] = {'count': 1, 'embeddings': [current_embedding]}
                else:
                    self.new_speaker_candidates['potential_new']['count'] += 1
                    self.new_speaker_candidates['potential_new']['embeddings'].append(current_embedding)
                
                # Create new speaker if enough evidence
                if self.new_speaker_candidates['potential_new']['count'] >= self.config.min_frames_for_new_speaker:
                    speaker_id = f"Speaker_{chr(65 + len(self.speaker_profiles))}"
                    avg_embedding = np.mean(self.new_speaker_candidates['potential_new']['embeddings'], axis=0)
                    self.speaker_profiles.append({'id': speaker_id, 'embedding': avg_embedding, 'count': 1})
                    self.speaker_count = len(self.speaker_profiles)
                    candidate_speaker = speaker_id
                    self.new_speaker_candidates.clear()
                else:
                    # Use closest existing speaker while collecting evidence  
                    best_idx = similarities.index(max_similarity)
                    candidate_speaker = self.speaker_profiles[best_idx]['id']
            else:
                # Force match to closest existing speaker
                best_idx = similarities.index(max_similarity)
                candidate_speaker = self.speaker_profiles[best_idx]['id']
                self.new_speaker_candidates.clear()
        
        # Stability control - avoid rapid switching
        if candidate_speaker != self.current_speaker:
            self.frames_since_change += 1
            if self.frames_since_change >= self.config.min_frames_for_change:
                self.current_speaker = candidate_speaker
                self.frames_since_change = 0
                self.speaker_changed = True
        else:
            self.frames_since_change = 0
            self.speaker_changed = False
        
        return self.current_speaker
    
    def update_speaker_count(self, audio_data: bytes, silence_frames: int = 0):
        """Process audio and update speaker"""
        self.identify_speaker(audio_data)
    
    def get_known_speakers(self) -> list:
        """Get list of known speaker IDs"""
        return [profile['id'] for profile in self.speaker_profiles]
    
    def get_current_features(self) -> Optional[Dict]:
        """Get current speaker features for database"""
        if self.audio_buffer:
            embedding = self._get_embedding(np.array(self.audio_buffer))
            return {f'feature_{i}': float(f) for i, f in enumerate(embedding)}
    
    def clear_feature_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer.clear()
    
    def has_speaker_changed(self) -> bool:
        """Check if speaker changed in last update"""
        return self.speaker_changed
    
    def reset_speakers(self):
        """Reset all speaker profiles"""
        self.speaker_profiles.clear()
        self.current_speaker = "Speaker_A"
        self.speaker_count = 1
        self.frames_since_change = 0
        self.audio_buffer.clear()
        self.new_speaker_candidates.clear()