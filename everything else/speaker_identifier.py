#!/usr/bin/env python3
"""Speaker Identification System with Training Capabilities"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque

class SpeakerIdentifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.speaker_names = {}  # Maps speaker IDs to actual names
        self.is_trained = False
        
    def extract_speaker_features(self, audio_data):
        """Extract comprehensive speaker features"""
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
        # Simplified MFCC calculation
        spectrum = np.abs(np.fft.fft(audio))
        mel_spectrum = np.log(spectrum[:len(spectrum)//2] + 1e-10)
        return np.mean(mel_spectrum[coefficient*10:(coefficient+1)*10])
    
    def _extract_formants(self, audio, formant_num):
        """Extract formant frequencies (simplified)"""
        # Simplified formant extraction
        spectrum = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio), 1/16000)
        positive_freqs = freqs[:len(freqs)//2]
        positive_spectrum = spectrum[:len(spectrum)//2]
        
        # Find peaks in spectrum (formants)
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
        # Simplified jitter calculation
        pitch_values = []
        for i in range(0, len(audio)-1024, 512):
            chunk = audio[i:i+1024]
            pitch = np.std(chunk)
            pitch_values.append(pitch)
        return np.std(pitch_values) if pitch_values else 0
    
    def _extract_shimmer(self, audio):
        """Extract shimmer (amplitude variation)"""
        # Simplified shimmer calculation
        amplitude_values = []
        for i in range(0, len(audio)-1024, 512):
            chunk = audio[i:i+1024]
            amplitude = np.mean(np.abs(chunk))
            amplitude_values.append(amplitude)
        return np.std(amplitude_values) if amplitude_values else 0
    
    def train_on_labeled_data(self, training_data):
        """
        Train the speaker identifier
        
        training_data format:
        [
            {
                'speaker_name': 'John',
                'audio_features': [feature_dict1, feature_dict2, ...],
                'audio_file': 'path/to/audio.wav'  # optional
            },
            {
                'speaker_name': 'Sarah',
                'audio_features': [feature_dict1, feature_dict2, ...],
                'audio_file': 'path/to/audio.wav'  # optional
            }
        ]
        """
        X = []  # Features
        y = []  # Speaker labels
        
        # Process training data
        for speaker_data in training_data:
            speaker_name = speaker_data['speaker_name']
            features_list = speaker_data['audio_features']
            
            # Add speaker to mapping
            if speaker_name not in self.speaker_names:
                self.speaker_names[speaker_name] = len(self.speaker_names)
            
            speaker_id = self.speaker_names[speaker_name]
            
            # Add features and labels
            for features in features_list:
                feature_vector = list(features.values())
                X.append(feature_vector)
                y.append(speaker_id)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        print(f"✅ Speaker identification model trained!")
        print(f"   Training accuracy: {train_score:.2f}")
        print(f"   Test accuracy: {test_score:.2f}")
        print(f"   Speakers learned: {list(self.speaker_names.keys())}")
        
        return train_score, test_score
    
    def identify_speaker(self, audio_features):
        """Identify speaker from audio features"""
        if not self.is_trained:
            return "Unknown", 0.0
        
        # Convert features to vector
        feature_vector = list(audio_features.values())
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Predict speaker
        speaker_id = self.classifier.predict(feature_vector_scaled)[0]
        confidence = np.max(self.classifier.predict_proba(feature_vector_scaled))
        
        # Get speaker name
        speaker_name = None
        for name, sid in self.speaker_names.items():
            if sid == speaker_id:
                speaker_name = name
                break
        
        return speaker_name, confidence
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'speaker_names': self.speaker_names,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.speaker_names = model_data['speaker_names']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   Speakers: {list(self.speaker_names.keys())}")
    
    def get_speaker_names(self):
        """Get list of known speaker names"""
        return list(self.speaker_names.keys())
    
    def add_speaker(self, speaker_name):
        """Add a new speaker to the model"""
        if speaker_name not in self.speaker_names:
            self.speaker_names[speaker_name] = len(self.speaker_names)
            print(f"✅ Added speaker: {speaker_name}")
        else:
            print(f"⚠️  Speaker {speaker_name} already exists") 