#!/usr/bin/env python3
"""Conditional Pipeline for routing transcribed text to Gemma"""

from typing import Dict, Any, Optional
from utils.utils import is_question, contains_keywords
from ai.gemma_client import GemmaClient

class ConditionalGemmaPipeline:
    """Routes text to Gemma based on conditions"""
    
    def __init__(self, model: str = "gemma3n:e4b", conditions: Optional[Dict[str, Any]] = None):
        self.model = model
        self.gemma_client = GemmaClient(model)
        
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
    
    def check_conditions(self, text: str, speaker_name: Optional[str] = None, 
                        emotion: Optional[str] = None, confidence: float = 0.0) -> bool:
        """Check if text meets conditions for routing to Gemma"""
        text = text.strip()
        
        # Basic length checks
        if not (self.conditions['min_length'] <= len(text) <= self.conditions['max_length']):
            return False
        
        # Check questions
        if self.conditions['questions'] and is_question(text):
            return True
        
        # Check keywords
        if contains_keywords(text, self.conditions['keywords']):
            return True
        
        # Check speakers (no emotion check)
        if (self.conditions['speakers'] and speaker_name in self.conditions['speakers'] 
            and confidence >= self.conditions['confidence_threshold']):
            return True
        
        return False
    
    def generate_response(self, text: str, speaker_name: Optional[str] = None, 
                         emotion: Optional[str] = None) -> Optional[str]:
        """Generate response using Gemma"""
        context = f"Speaker: {speaker_name or 'Unknown'}"
        return self.gemma_client.generate_response(text, context)
    
    def process_transcription(self, text: str, speaker_name: Optional[str] = None, 
                            emotion: Optional[str] = None, confidence: float = 0.0) -> Optional[str]:
        """Process transcribed text and conditionally route to Gemma"""
        print(f"🎯 Processing: '{text}' | Speaker: {speaker_name or 'Unknown'}")
        
        if self.check_conditions(text, speaker_name, emotion, confidence):
            print("🤖 Routing to Gemma...")
            response = self.generate_response(text, speaker_name, emotion)
            if response:
                print(f"💬 Gemma: {response}")
            return response
        else:
            print("⏭️  Conditions not met - skipping Gemma")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        pass  # No cleanup needed for client

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