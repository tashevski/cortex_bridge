#!/usr/bin/env python3
"""Conditional Pipeline for routing transcribed text to Gemma - No Emotion Classifier"""

import json
import requests
from gemma_runner import GemmaRunner

class ConditionalGemmaPipeline:
    def __init__(self, model="gemma3n:e4b", conditions=None):
        self.model = model
        self.gemma_runner = None
        self.is_initialized = False
        
        # Default conditions - NO EMOTION CLASSIFIER
        self.conditions = conditions or {
            'questions': True,
            'keywords': ['help', 'explain', 'what', 'how', 'why'],
            'emotions': [],  # Empty - no emotion classification
            'speakers': None,
            'confidence_threshold': 0.7,
            'min_length': 5,
            'max_length': 500
        }
    
    def initialize_gemma(self):
        """Initialize Gemma runner"""
        if not self.is_initialized:
            self.gemma_runner = GemmaRunner(self.model)
            self.gemma_runner.start_server()
            self.gemma_runner.pull_model()
            self.is_initialized = True
    
    def check_conditions(self, text, speaker_name=None, emotion=None, confidence=0.0):
        """Check if text meets conditions for routing to Gemma"""
        text = text.strip()
        
        # Basic length checks
        if not (self.conditions['min_length'] <= len(text) <= self.conditions['max_length']):
            return False
        
        # Check questions
        if self.conditions['questions'] and self._is_question(text):
            return True
        
        # Check keywords
        if any(keyword.lower() in text.lower() for keyword in self.conditions['keywords']):
            return True
        
        # Check speakers (no emotion check)
        if (self.conditions['speakers'] and speaker_name in self.conditions['speakers'] 
            and confidence >= self.conditions['confidence_threshold']):
            return True
        
        return False
    
    def _is_question(self, text):
        """Check if text is a question"""
        if '?' in text:
            return True
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        text_lower = text.lower().strip()
        
        return (any(text_lower.startswith(word + ' ') for word in question_words) or
                any(text_lower.startswith(prefix) for prefix in ['is ', 'are ', 'do ', 'does ', 'can ', 'will ']))
    
    def generate_response(self, text, speaker_name=None, emotion=None):
        """Generate response using Gemma"""
        if not self.is_initialized:
            self.initialize_gemma()
        
        # Create context-aware prompt
        context = f"Speaker: {speaker_name or 'Unknown'}"
        
        prompt = f"Context: {context}\nUser: {text}\n\nPlease provide a helpful, concise response:"
        
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                json={'model': self.model, 'prompt': prompt.strip(), 'stream': False}, timeout=30)
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    
    def process_transcription(self, text, speaker_name=None, emotion=None, confidence=0.0):
        """Process transcribed text and conditionally route to Gemma"""
        print(f"🎯 Processing: '{text}' | Speaker: {speaker_name or 'Unknown'}")
        
        if self.check_conditions(text, speaker_name, emotion, confidence):
            print("🤖 Routing to Gemma...")
            response = self.generate_response(text, speaker_name, emotion)
            print(f"💬 Gemma: {response}")
            return response
        else:
            print("⏭️  Conditions not met - skipping Gemma")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.gemma_runner:
            self.gemma_runner.cleanup()

# Pre-configured condition sets - NO EMOTION CLASSIFIER
CONDITIONS = {
    'questions_only': {
        'questions': True, 'keywords': [], 'emotions': [], 'speakers': None,
        'confidence_threshold': 0.7, 'min_length': 5, 'max_length': 500
    },
    'keywords_only': {
        'questions': False, 'keywords': ['help', 'explain', 'what', 'how', 'why'],
        'emotions': [], 'speakers': None,
        'confidence_threshold': 0.6, 'min_length': 5, 'max_length': 500
    },
    'route_all': {
        'questions': True, 'keywords': [], 'emotions': [], 'speakers': None,
        'confidence_threshold': 0.0, 'min_length': 1, 'max_length': 1000
    }
}