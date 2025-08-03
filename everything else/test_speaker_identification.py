#!/usr/bin/env python3
"""Test script for speaker identification system"""

import json
import numpy as np
from speaker_identifier import SpeakerIdentifier

def create_demo_training_data():
    """Create demo training data for testing"""
    # Simulate different voice characteristics for different speakers
    demo_data = [
        {
            'speaker_name': 'John',
            'audio_features': [
                {
                    'energy': 1500.0, 'pitch_estimate': 120.0, 'zero_crossings': 45,
                    'spectral_centroid': 800.0, 'energy_variance': 200.0,
                    'peak_amplitude': 3000.0, 'rms_energy': 1200.0,
                    'mfcc_1': 0.5, 'mfcc_2': 0.3, 'mfcc_3': 0.2,
                    'formant_1': 500.0, 'formant_2': 1500.0,
                    'jitter': 0.02, 'shimmer': 0.03
                },
                {
                    'energy': 1600.0, 'pitch_estimate': 125.0, 'zero_crossings': 48,
                    'spectral_centroid': 820.0, 'energy_variance': 220.0,
                    'peak_amplitude': 3200.0, 'rms_energy': 1250.0,
                    'mfcc_1': 0.52, 'mfcc_2': 0.32, 'mfcc_3': 0.22,
                    'formant_1': 520.0, 'formant_2': 1550.0,
                    'jitter': 0.021, 'shimmer': 0.031
                }
            ]
        },
        {
            'speaker_name': 'Sarah',
            'audio_features': [
                {
                    'energy': 2000.0, 'pitch_estimate': 180.0, 'zero_crossings': 55,
                    'spectral_centroid': 1000.0, 'energy_variance': 300.0,
                    'peak_amplitude': 4000.0, 'rms_energy': 1600.0,
                    'mfcc_1': 0.7, 'mfcc_2': 0.5, 'mfcc_3': 0.3,
                    'formant_1': 600.0, 'formant_2': 1800.0,
                    'jitter': 0.025, 'shimmer': 0.035
                },
                {
                    'energy': 2100.0, 'pitch_estimate': 185.0, 'zero_crossings': 58,
                    'spectral_centroid': 1020.0, 'energy_variance': 320.0,
                    'peak_amplitude': 4200.0, 'rms_energy': 1650.0,
                    'mfcc_1': 0.72, 'mfcc_2': 0.52, 'mfcc_3': 0.32,
                    'formant_1': 620.0, 'formant_2': 1850.0,
                    'jitter': 0.026, 'shimmer': 0.036
                }
            ]
        },
        {
            'speaker_name': 'Mike',
            'audio_features': [
                {
                    'energy': 1200.0, 'pitch_estimate': 100.0, 'zero_crossings': 35,
                    'spectral_centroid': 600.0, 'energy_variance': 150.0,
                    'peak_amplitude': 2500.0, 'rms_energy': 1000.0,
                    'mfcc_1': 0.3, 'mfcc_2': 0.2, 'mfcc_3': 0.1,
                    'formant_1': 400.0, 'formant_2': 1200.0,
                    'jitter': 0.018, 'shimmer': 0.025
                },
                {
                    'energy': 1250.0, 'pitch_estimate': 105.0, 'zero_crossings': 38,
                    'spectral_centroid': 620.0, 'energy_variance': 160.0,
                    'peak_amplitude': 2600.0, 'rms_energy': 1050.0,
                    'mfcc_1': 0.32, 'mfcc_2': 0.22, 'mfcc_3': 0.12,
                    'formant_1': 420.0, 'formant_2': 1250.0,
                    'jitter': 0.019, 'shimmer': 0.026
                }
            ]
        }
    ]
    
    return demo_data

def test_speaker_identification():
    """Test the speaker identification system"""
    print("🧪 Testing Speaker Identification System")
    print("=" * 50)
    
    # Create speaker identifier
    speaker_id = SpeakerIdentifier()
    
    # Create demo training data
    training_data = create_demo_training_data()
    
    print("📊 Training data created:")
    for speaker_data in training_data:
        print(f"   {speaker_data['speaker_name']}: {len(speaker_data['audio_features'])} samples")
    
    # Train the model
    print("\n🤖 Training speaker identification model...")
    train_score, test_score = speaker_id.train_on_labeled_data(training_data)
    
    # Test identification with known speakers
    print("\n🔍 Testing speaker identification...")
    
    # Test with John's voice characteristics
    john_features = {
        'energy': 1550.0, 'pitch_estimate': 122.0, 'zero_crossings': 46,
        'spectral_centroid': 810.0, 'energy_variance': 210.0,
        'peak_amplitude': 3100.0, 'rms_energy': 1225.0,
        'mfcc_1': 0.51, 'mfcc_2': 0.31, 'mfcc_3': 0.21,
        'formant_1': 510.0, 'formant_2': 1525.0,
        'jitter': 0.0205, 'shimmer': 0.0305
    }
    
    speaker_name, confidence = speaker_id.identify_speaker(john_features)
    print(f"   John's voice → {speaker_name} (confidence: {confidence:.2f})")
    
    # Test with Sarah's voice characteristics
    sarah_features = {
        'energy': 2050.0, 'pitch_estimate': 182.0, 'zero_crossings': 56,
        'spectral_centroid': 1010.0, 'energy_variance': 310.0,
        'peak_amplitude': 4100.0, 'rms_energy': 1625.0,
        'mfcc_1': 0.71, 'mfcc_2': 0.51, 'mfcc_3': 0.31,
        'formant_1': 610.0, 'formant_2': 1825.0,
        'jitter': 0.0255, 'shimmer': 0.0355
    }
    
    speaker_name, confidence = speaker_id.identify_speaker(sarah_features)
    print(f"   Sarah's voice → {speaker_name} (confidence: {confidence:.2f})")
    
    # Test with Mike's voice characteristics
    mike_features = {
        'energy': 1225.0, 'pitch_estimate': 102.0, 'zero_crossings': 36,
        'spectral_centroid': 610.0, 'energy_variance': 155.0,
        'peak_amplitude': 2550.0, 'rms_energy': 1025.0,
        'mfcc_1': 0.31, 'mfcc_2': 0.21, 'mfcc_3': 0.11,
        'formant_1': 410.0, 'formant_2': 1225.0,
        'jitter': 0.0185, 'shimmer': 0.0255
    }
    
    speaker_name, confidence = speaker_id.identify_speaker(mike_features)
    print(f"   Mike's voice → {speaker_name} (confidence: {confidence:.2f})")
    
    # Test with unknown voice (should have low confidence)
    unknown_features = {
        'energy': 3000.0, 'pitch_estimate': 200.0, 'zero_crossings': 70,
        'spectral_centroid': 1500.0, 'energy_variance': 500.0,
        'peak_amplitude': 6000.0, 'rms_energy': 2500.0,
        'mfcc_1': 1.0, 'mfcc_2': 0.8, 'mfcc_3': 0.6,
        'formant_1': 800.0, 'formant_2': 2200.0,
        'jitter': 0.04, 'shimmer': 0.05
    }
    
    speaker_name, confidence = speaker_id.identify_speaker(unknown_features)
    print(f"   Unknown voice → {speaker_name} (confidence: {confidence:.2f})")
    
    # Save the model
    speaker_id.save_model("demo_speaker_identification_model.pkl")
    
    print(f"\n✅ Test completed!")
    print(f"   Model saved as: demo_speaker_identification_model.pkl")
    print(f"   Training accuracy: {train_score:.2f}")
    print(f"   Test accuracy: {test_score:.2f}")

if __name__ == "__main__":
    test_speaker_identification() 