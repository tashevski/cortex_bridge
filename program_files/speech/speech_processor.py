#!/usr/bin/env python3
"""Speech processing and speaker detection"""

import numpy as np
import webrtcvad
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
    """Minimal ECAPA-TDNN based speaker detection"""
    
    def __init__(self, sample_rate: int = 16000, enhanced_db=None, speaker_clustering=None, max_speakers: int = 8):
        self.sample_rate = sample_rate
        self.current_speaker = "Speaker_A"
        self.speaker_count = 1
        self.speaker_profiles = []
        self.audio_buffer = []
        self.buffer_size = 16000  # 1.0 seconds for stable embeddings
        self.similarity_threshold = 0.40  # Much lower threshold to detect different speakers
        self.frames_since_change = 0
        self.min_frames_for_change = 4    # Balance between stability and responsiveness
        self.min_speech_energy = 0.02  # Minimum energy to process speaker detection (increased)
        self.max_speakers = max_speakers  # Configurable max speakers
        self.new_speaker_candidates = {}  # Track potential new speakers
        self.min_frames_for_new_speaker = 15  # Require many consistent frames before creating new speaker
        self.speaker_changed = False  # Flag to track speaker changes
        self.previous_speaker = None  # Track previous speaker for change detection
        
        # Load ECAPA-TDNN model
        try:
            import speechbrain.pretrained
            self.speaker_model = speechbrain.pretrained.EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/spkrec-ecapa-voxceleb"
            )
            print("✅ Using ECAPA-TDNN embeddings")
        except (ImportError, OSError, Exception) as e:
            # Fallback to spectral features if SpeechBrain not available or has issues
            self.speaker_model = None
            print(f"⚠️ SpeechBrain not available ({type(e).__name__}) - using spectral features")
    
    def extract_features(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract discriminative spectral features for speaker identification"""
        # Compute FFT and get magnitude spectrum
        fft = np.fft.rfft(audio_np, n=1024)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(1024, 1/self.sample_rate)
        
        # Focus on voice frequency range (80-4000 Hz) - more discriminative
        voice_mask = (freqs >= 80) & (freqs <= 4000)
        voice_magnitude = magnitude[voice_mask]
        voice_freqs = freqs[voice_mask]
        
        # Extract 8 frequency bands in voice range for better discrimination  
        bands = np.array_split(voice_magnitude, 8)
        band_energies = [np.mean(band) for band in bands]
        
        # Add voice-specific features
        if np.sum(voice_magnitude) > 0:
            # Spectral centroid in voice range
            centroid = np.sum(voice_freqs * voice_magnitude) / np.sum(voice_magnitude)
            # Spectral spread (bandwidth)
            spread = np.sqrt(np.sum(((voice_freqs - centroid) ** 2) * voice_magnitude) / np.sum(voice_magnitude))
            # Spectral skewness (asymmetry)
            skewness = np.sum(((voice_freqs - centroid) ** 3) * voice_magnitude) / (np.sum(voice_magnitude) * spread ** 3) if spread > 0 else 0
        else:
            centroid = spread = skewness = 0
            
        return np.array(band_energies + [centroid, spread, skewness])
    
    def get_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract ECAPA-TDNN embedding or fallback to spectral features"""
        if self.speaker_model is not None:
            # Use ECAPA-TDNN for embedding extraction
            import torch
            # Ensure audio is normalized
            audio_normalized = audio_np / (np.max(np.abs(audio_np)) + 1e-10)
            audio_tensor = torch.tensor(audio_normalized, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding_np = embedding.squeeze().cpu().numpy()
                # L2 normalize the embedding
                embedding_norm = np.linalg.norm(embedding_np)
                if embedding_norm > 0:
                    embedding_np = embedding_np / embedding_norm
                return embedding_np
        else:
            # Fallback to spectral features
            return self.extract_features(audio_np)
    
    def identify_speaker(self, audio_data: bytes) -> str:
        """Real-time speaker identification using ECAPA-TDNN"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) == 0:
            return self.current_speaker
            
        # Add to rolling buffer
        self.audio_buffer.extend(audio_np)
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # Need minimum audio for reliable embeddings
        if len(self.audio_buffer) < self.buffer_size:
            return self.current_speaker
            
        # Check if there's enough speech energy (avoid processing silence/noise)
        buffer_array = np.array(self.audio_buffer)
        energy = np.sqrt(np.mean(buffer_array ** 2))
        
        # Also check peak amplitude to catch brief loud noises
        peak_amplitude = np.max(np.abs(buffer_array))
        
        if energy < self.min_speech_energy or peak_amplitude < 0.05:
            # Clear any pending new speaker candidates during silence
            self.new_speaker_candidates.clear()
            return self.current_speaker
            
        # Extract embedding from current buffer
        current_embedding = self.get_embedding(np.array(self.audio_buffer))
        
        # First speaker - but only if there's sufficient energy
        if not self.speaker_profiles:
            # Double-check energy for first speaker to avoid creating from silence
            if energy >= self.min_speech_energy * 2:  # Higher threshold for first speaker
                self.speaker_profiles.append({
                    'id': 'Speaker_A', 
                    'embedding': current_embedding,
                    'count': 1
                })
                self.speaker_count = 1
                return 'Speaker_A'
            else:
                return self.current_speaker  # Stay silent until real speech
        
        # Compare with existing speakers using cosine similarity
        # (embeddings are already L2-normalized in get_embedding)
        similarities = []
        for profile in self.speaker_profiles:
            similarity = np.dot(current_embedding, profile['embedding'])
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        
        # Debug output (disabled)
        # if len(similarities) >= 1:
        #     speakers_str = ', '.join([f"{self.speaker_profiles[i]['id']}:{similarities[i]:.3f}" for i in range(len(similarities))])
        #     print(f"🔍 Energy: {energy:.3f}, Speakers: [{speakers_str}], Max: {max_similarity:.3f}, Threshold: {self.similarity_threshold}")
        #     if max_similarity < self.similarity_threshold:
        #         print(f"⚠️  Below threshold - candidate count: {self.new_speaker_candidates.get('potential_new', {}).get('count', 0)}")
        
        if max_similarity >= self.similarity_threshold:
            # Match found - update embedding with exponential moving average
            best_idx = similarities.index(max_similarity)
            profile = self.speaker_profiles[best_idx]
            
            alpha = 0.05  # Lower learning rate for more stable profiles
            profile['embedding'] = (1 - alpha) * profile['embedding'] + alpha * current_embedding
            profile['count'] += 1
            
            candidate_speaker = profile['id']
            # Clear any pending new speaker candidates since we matched existing
            self.new_speaker_candidates.clear()
        else:
            # Potential new speaker - need consistent detection
            if len(self.speaker_profiles) < self.max_speakers:
                # Find which existing speaker is closest (even if below threshold)
                best_idx = similarities.index(max_similarity)
                closest_speaker = self.speaker_profiles[best_idx]['id']
                
                # Track this as a potential new speaker
                if 'potential_new' not in self.new_speaker_candidates:
                    self.new_speaker_candidates['potential_new'] = {
                        'count': 1,
                        'embeddings': [current_embedding],
                        'closest_speaker': closest_speaker
                    }
                else:
                    self.new_speaker_candidates['potential_new']['count'] += 1
                    self.new_speaker_candidates['potential_new']['embeddings'].append(current_embedding)
                
                # If we've seen this potential new speaker enough times, create it
                if self.new_speaker_candidates['potential_new']['count'] >= self.min_frames_for_new_speaker:
                    speaker_id = f"Speaker_{chr(65 + len(self.speaker_profiles))}"
                    # Use average of collected embeddings for stability
                    avg_embedding = np.mean(self.new_speaker_candidates['potential_new']['embeddings'], axis=0)
                    self.speaker_profiles.append({
                        'id': speaker_id,
                        'embedding': avg_embedding,
                        'count': self.new_speaker_candidates['potential_new']['count']
                    })
                    self.speaker_count = len(self.speaker_profiles)
                    candidate_speaker = speaker_id
                    # Clear the candidate tracker
                    self.new_speaker_candidates.clear()
                else:
                    # Still collecting evidence - use closest existing speaker
                    candidate_speaker = closest_speaker
            else:
                # Force match to closest existing speaker
                best_idx = similarities.index(max_similarity)
                candidate_speaker = self.speaker_profiles[best_idx]['id']
                # Clear any pending new speaker candidates
                self.new_speaker_candidates.clear()
        
        # Stability control - avoid rapid switching
        if candidate_speaker != self.current_speaker:
            self.frames_since_change += 1
            if self.frames_since_change >= self.min_frames_for_change:
                self.previous_speaker = self.current_speaker
                self.current_speaker = candidate_speaker
                self.frames_since_change = 0
                self.speaker_changed = True
                print(f"\n🎤 Speaker changed: {self.previous_speaker} → {self.current_speaker}\n")
        else:
            self.frames_since_change = 0
            self.speaker_changed = False
            
        return self.current_speaker
    
    def update_speaker_count(self, audio_data: bytes, silence_frames: int = 0):
        """Process audio frame and update current speaker"""
        new_speaker = self.identify_speaker(audio_data)
        if new_speaker != self.current_speaker:
            self.current_speaker = new_speaker
    
    def get_known_speakers(self) -> list:
        """Get list of known speakers"""
        return [profile['id'] for profile in self.speaker_profiles]
    
    def get_current_features(self) -> Optional[Dict]:
        """Get current speaker features for database storage"""
        if self.audio_buffer:
            embedding = self.get_embedding(np.array(self.audio_buffer))
            return {f'feature_{i}': float(f) for i, f in enumerate(embedding[:6])}  # First 6 dimensions
    
    def clear_feature_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()
    
    def has_speaker_changed(self) -> bool:
        """Check if speaker changed in the last update"""
        return self.speaker_changed
    
    def reset_speakers(self):
        """Reset all speaker profiles for new conversation"""
        self.speaker_profiles.clear()
        self.current_speaker = "Speaker_A"
        self.speaker_count = 1
        self.frames_since_change = 0
        self.audio_buffer.clear()
        self.new_speaker_candidates.clear() 