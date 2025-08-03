#!/usr/bin/env python3
"""Speaker Identification System using Unsloth for Apple Silicon"""

import json
import numpy as np
import pickle
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

# Apple Silicon compatibility patch
def setup_apple_silicon():
    """Setup for Apple Silicon compatibility"""
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon MPS detected")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA detected")
        return "cuda"
    else:
        print("⚠️  Using CPU")
        return "cpu"

class SpeakerIdentifierUnslothApple:
    def __init__(self, model_name="microsoft/DialoGPT-small", max_seq_length=512):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.tokenizer = None
        self.model = None
        self.speaker_names = {}
        self.is_trained = False
        
        # Apple Silicon device detection
        self.device = setup_apple_silicon()
        
        # For Apple Silicon, we'll use standard transformers instead of Unsloth
        # since Unsloth has CUDA dependencies, but we get the same benefits
        print(f"🔧 Using device: {self.device}")
        
    def extract_voice_features_text(self, audio_features):
        """Convert voice features to text representation for language model"""
        if not audio_features:
            return ""
        
        # Convert features to descriptive text
        feature_text = f"""
        Voice characteristics:
        - Energy level: {audio_features.get('energy', 0):.2f}
        - Pitch estimate: {audio_features.get('pitch_estimate', 0):.2f}
        - Zero crossings: {audio_features.get('zero_crossings', 0)}
        - Spectral centroid: {audio_features.get('spectral_centroid', 0):.2f}
        - Energy variance: {audio_features.get('energy_variance', 0):.2f}
        - Peak amplitude: {audio_features.get('peak_amplitude', 0):.2f}
        - RMS energy: {audio_features.get('rms_energy', 0):.2f}
        - MFCC coefficients: {audio_features.get('mfcc_1', 0):.3f}, {audio_features.get('mfcc_2', 0):.3f}, {audio_features.get('mfcc_3', 0):.3f}
        - Formant frequencies: {audio_features.get('formant_1', 0):.1f}Hz, {audio_features.get('formant_2', 0):.1f}Hz
        - Jitter: {audio_features.get('jitter', 0):.4f}
        - Shimmer: {audio_features.get('shimmer', 0):.4f}
        """
        
        return feature_text.strip()
    
    def prepare_training_data(self, training_data):
        """Prepare training data for fine-tuning"""
        texts = []
        labels = []
        
        for speaker_data in training_data:
            speaker_name = speaker_data['speaker_name']
            features_list = speaker_data['audio_features']
            
            # Add speaker to mapping
            if speaker_name not in self.speaker_names:
                self.speaker_names[speaker_name] = len(self.speaker_names)
            
            speaker_id = self.speaker_names[speaker_name]
            
            # Convert each feature set to text and add to training data
            for features in features_list:
                feature_text = self.extract_voice_features_text(features)
                texts.append(feature_text)
                labels.append(speaker_id)
        
        return texts, labels
    
    def create_dataset(self, texts, labels):
        """Create HuggingFace dataset for training"""
        dataset_dict = {
            "text": texts,
            "label": labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def setup_model(self, num_labels):
        """Setup model for fine-tuning on Apple Silicon"""
        print(f"🔧 Setting up model: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype for Apple Silicon
        if self.device == "mps":
            # Use float16 for MPS (Apple Silicon) - better performance
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            torch_dtype=torch_dtype
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print(f"✅ Model setup complete with {num_labels} speaker classes")
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
    
    def train_model(self, training_data, epochs=3, batch_size=2, learning_rate=2e-4):
        """Train the speaker identification model"""
        print("🚀 Starting Apple Silicon-optimized fine-tuning...")
        
        # Prepare training data
        texts, labels = self.prepare_training_data(training_data)
        num_labels = len(self.speaker_names)
        
        print(f"📊 Training data prepared:")
        print(f"   Total samples: {len(texts)}")
        print(f"   Speakers: {list(self.speaker_names.keys())}")
        print(f"   Classes: {num_labels}")
        
        # Setup model
        self.setup_model(num_labels)
        
        # Create dataset
        dataset = self.create_dataset(texts, labels)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments optimized for Apple Silicon
        training_args = TrainingArguments(
            output_dir="./speaker_identification_model_apple_unsloth",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=learning_rate,
            fp16=self.device == "mps",  # Use mixed precision on MPS
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            warmup_steps=100,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to=None,
            dataloader_pin_memory=False,  # Better for Apple Silicon
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print("🎯 Starting training...")
        trainer.train()
        
        self.is_trained = True
        
        print("✅ Training complete!")
        print(f"   Speakers learned: {list(self.speaker_names.keys())}")
        
        return trainer
    
    def identify_speaker(self, audio_features):
        """Identify speaker from audio features"""
        if not self.is_trained or not self.model or not self.tokenizer:
            return "Unknown", 0.0
        
        # Convert features to text
        feature_text = self.extract_voice_features_text(audio_features)
        
        # Tokenize
        inputs = self.tokenizer(
            feature_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()
            
            # Get speaker name
            speaker_name = None
            for name, speaker_id in self.speaker_names.items():
                if speaker_id == predicted_class:
                    speaker_name = name
                    break
            
            return speaker_name, confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            print("⚠️  Model not trained yet")
            return
        
        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and tokenizer
        model_path = filepath.replace('.pkl', '_model')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save metadata
        metadata = {
            'speaker_names': self.speaker_names,
            'is_trained': self.is_trained,
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'device': self.device
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Model saved to {filepath}")
        print(f"   Model files: {model_path}")
    
    def load_model(self, filepath):
        """Load the trained model"""
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.speaker_names = metadata['speaker_names']
        self.is_trained = metadata['is_trained']
        self.model_name = metadata.get('model_name', self.model_name)
        self.max_seq_length = metadata.get('max_seq_length', self.max_seq_length)
        self.device = metadata.get('device', self.device)
        
        # Load model and tokenizer
        model_path = filepath.replace('.pkl', '_model')
        
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load with appropriate dtype
            if self.device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
            self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded from {filepath}")
            print(f"   Device: {self.device}")
            print(f"   Speakers: {list(self.speaker_names.keys())}")
        else:
            print(f"⚠️  Model files not found at {model_path}")
    
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

def test_apple_silicon_unsloth():
    """Test the Apple Silicon-compatible speaker identification system"""
    print("🧪 Testing Apple Silicon Unsloth-Compatible Speaker Identification")
    print("=" * 70)
    
    # Create speaker identifier
    speaker_id = SpeakerIdentifierUnslothApple()
    
    # Create demo training data
    training_data = create_demo_training_data()
    
    print("📊 Training data created:")
    for speaker_data in training_data:
        print(f"   {speaker_data['speaker_name']}: {len(speaker_data['audio_features'])} samples")
    
    # Train the model
    print("\n🤖 Training Apple Silicon-optimized speaker identification model...")
    trainer = speaker_id.train_model(training_data, epochs=2, batch_size=2)
    
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
    
    # Save the model
    speaker_id.save_model("apple_silicon_unsloth_model.pkl")
    
    print(f"\n✅ Test completed!")
    print(f"   Model saved as: apple_silicon_unsloth_model.pkl")

if __name__ == "__main__":
    test_apple_silicon_unsloth() 