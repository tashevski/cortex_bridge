#!/usr/bin/env python3
"""Collect training data for Apple Silicon Unsloth-compatible speaker identification"""

import json
import numpy as np
import pyaudio
import wave
from pathlib import Path
from speaker_identifier_unsloth_apple import SpeakerIdentifierUnslothApple

class AppleUnslothTrainingDataCollector:
    def __init__(self):
        self.speaker_identifier = SpeakerIdentifierUnslothApple()
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_size = 2048
        
    def extract_voice_features(self, audio_data):
        """Extract voice features from audio data"""
        if len(audio_data) == 0:
            return None
            
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Basic features
        features = {
            'energy': np.mean(np.abs(audio_np)),
            'pitch_estimate': np.std(audio_np),
            'zero_crossings': np.sum(np.diff(np.sign(audio_np)) != 0),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2])),
            'energy_variance': np.var(np.abs(audio_np)),
            'peak_amplitude': np.max(np.abs(audio_np)),
            'rms_energy': np.sqrt(np.mean(audio_np**2))
        }
        
        # Advanced features for speaker identification
        features.update({
            'mfcc_1': self._extract_mfcc(audio_np, 1),
            'mfcc_2': self._extract_mfcc(audio_np, 2),
            'mfcc_3': self._extract_mfcc(audio_np, 3),
            'formant_1': self._extract_formants(audio_np, 1),
            'formant_2': self._extract_formants(audio_np, 2),
            'jitter': self._extract_jitter(audio_np),
            'shimmer': self._extract_shimmer(audio_np)
        })
        
        return features
    
    def _extract_mfcc(self, audio, coefficient):
        """Extract MFCC coefficients (simplified)"""
        spectrum = np.abs(np.fft.fft(audio))
        mel_spectrum = np.log(spectrum[:len(spectrum)//2] + 1e-10)
        return np.mean(mel_spectrum[coefficient*10:(coefficient+1)*10])
    
    def _extract_formants(self, audio, formant_num):
        """Extract formant frequencies (simplified)"""
        spectrum = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio), 1/16000)
        positive_freqs = freqs[:len(freqs)//2]
        positive_spectrum = spectrum[:len(spectrum)//2]
        
        peaks = []
        for i in range(1, len(positive_spectrum)-1):
            if positive_spectrum[i] > positive_spectrum[i-1] and positive_spectrum[i] > positive_spectrum[i+1]:
                peaks.append((positive_freqs[i], positive_spectrum[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        if len(peaks) >= formant_num:
            return peaks[formant_num-1][0]
        return 0
    
    def _extract_jitter(self, audio):
        """Extract jitter (pitch variation)"""
        pitch_values = []
        for i in range(0, len(audio)-1024, 512):
            chunk = audio[i:i+1024]
            pitch = np.std(chunk)
            pitch_values.append(pitch)
        return np.std(pitch_values) if pitch_values else 0
    
    def _extract_shimmer(self, audio):
        """Extract shimmer (amplitude variation)"""
        amplitude_values = []
        for i in range(0, len(audio)-1024, 512):
            chunk = audio[i:i+1024]
            amplitude = np.mean(np.abs(chunk))
            amplitude_values.append(amplitude)
        return np.std(amplitude_values) if amplitude_values else 0
        
    def record_speaker_sample(self, speaker_name, duration=5, filename=None):
        """Record a sample from a speaker"""
        print(f"🎤 Recording sample for {speaker_name}...")
        print(f"   Speak for {duration} seconds...")
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        for i in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Extract features from the recording
        audio_data = b''.join(frames)
        features = self.extract_voice_features(audio_data)
        
        # Save audio file if filename provided
        if filename:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
        
        print(f"✅ Recorded sample for {speaker_name}")
        return features
    
    def collect_training_data(self, speakers_config):
        """
        Collect training data for multiple speakers
        
        speakers_config format:
        {
            'John': {'samples': 10, 'duration': 5},
            'Sarah': {'samples': 10, 'duration': 5},
            'Mike': {'samples': 10, 'duration': 5}
        }
        """
        training_data = []
        
        for speaker_name, config in speakers_config.items():
            print(f"\n🎤 Collecting data for {speaker_name}")
            print(f"   Samples: {config['samples']}")
            print(f"   Duration: {config['duration']} seconds each")
            
            speaker_features = []
            
            for i in range(config['samples']):
                print(f"\n   Sample {i+1}/{config['samples']}")
                
                # Create filename
                filename = f"training_data_apple_unsloth/{speaker_name}_sample_{i+1}.wav"
                Path("training_data_apple_unsloth").mkdir(exist_ok=True)
                
                # Record sample
                features = self.record_speaker_sample(
                    speaker_name, 
                    config['duration'], 
                    filename
                )
                
                if features:
                    speaker_features.append(features)
                
                input("Press Enter to continue to next sample...")
            
            training_data.append({
                'speaker_name': speaker_name,
                'audio_features': speaker_features
            })
        
        return training_data
    
    def save_training_data(self, training_data, filepath):
        """Save training data to file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for speaker_data in training_data:
            serializable_features = []
            for features in speaker_data['audio_features']:
                serializable_features.append({k: float(v) for k, v in features.items()})
            
            serializable_data.append({
                'speaker_name': speaker_data['speaker_name'],
                'audio_features': serializable_features
            })
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Training data saved to {filepath}")
    
    def load_training_data(self, filepath):
        """Load training data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        training_data = []
        for speaker_data in data:
            features_list = []
            for features in speaker_data['audio_features']:
                features_list.append(features)
            
            training_data.append({
                'speaker_name': speaker_data['speaker_name'],
                'audio_features': features_list
            })
        
        return training_data

def main():
    """Main training data collection script for Apple Silicon Unsloth"""
    collector = AppleUnslothTrainingDataCollector()
    
    # Configure speakers to train
    speakers_config = {
        'John': {'samples': 5, 'duration': 3},
        'Sarah': {'samples': 5, 'duration': 3},
        'Mike': {'samples': 5, 'duration': 3}
    }
    
    print("🎤 Apple Silicon Unsloth-Compatible Speaker Identification Training")
    print("=" * 70)
    
    # Collect training data
    training_data = collector.collect_training_data(speakers_config)
    
    # Save training data
    collector.save_training_data(training_data, "apple_unsloth_speaker_training_data.json")
    
    # Train the model
    print("\n🤖 Training Apple Silicon-optimized speaker identification model...")
    trainer = collector.speaker_identifier.train_model(training_data, epochs=2, batch_size=2)
    
    # Save the trained model
    collector.speaker_identifier.save_model("apple_silicon_unsloth_model.pkl")
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved as: apple_silicon_unsloth_model.pkl")
    print(f"   Training data saved as: apple_unsloth_speaker_training_data.json")

if __name__ == "__main__":
    main() 