#!/usr/bin/env python3
"""Collect training data for speaker identification"""

import json
import numpy as np
import pyaudio
import wave
from pathlib import Path
from speaker_identifier import SpeakerIdentifier

class TrainingDataCollector:
    def __init__(self):
        self.speaker_identifier = SpeakerIdentifier()
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_size = 2048
        
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
        features = self.speaker_identifier.extract_speaker_features(audio_data)
        
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
                filename = f"training_data/{speaker_name}_sample_{i+1}.wav"
                Path("training_data").mkdir(exist_ok=True)
                
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
    """Main training data collection script"""
    collector = TrainingDataCollector()
    
    # Configure speakers to train
    speakers_config = {
        'John': {'samples': 5, 'duration': 3},
        'Sarah': {'samples': 5, 'duration': 3},
        'Mike': {'samples': 5, 'duration': 3}
    }
    
    print("🎤 Speaker Identification Training Data Collection")
    print("=" * 50)
    
    # Collect training data
    training_data = collector.collect_training_data(speakers_config)
    
    # Save training data
    collector.save_training_data(training_data, "speaker_training_data.json")
    
    # Train the model
    print("\n🤖 Training speaker identification model...")
    train_score, test_score = collector.speaker_identifier.train_on_labeled_data(training_data)
    
    # Save the trained model
    collector.speaker_identifier.save_model("speaker_identification_model.pkl")
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved as: speaker_identification_model.pkl")
    print(f"   Training data saved as: speaker_training_data.json")

if __name__ == "__main__":
    main() 