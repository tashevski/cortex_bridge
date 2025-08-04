#!/usr/bin/env python3
"""Smart model selection to minimize loading overhead"""

import time
from typing import Optional
from ai.gemma_client import GemmaClient

class SmartModelSelector:
    """Intelligently selects models to minimize loading overhead"""
    
    def __init__(self):
        self.current_model = None
        self.last_switch_time = 0
        self.switch_threshold = 30  # seconds before considering switch
        self.complexity_keywords = {
            'complex': ['analyze', 'explain', 'reasoning', 'complex', 'detailed', 'comprehensive'],
            'simple': ['what', 'when', 'where', 'yes', 'no', 'quick', 'simple']
        }
    
    def should_use_e4b(self, prompt: str, context: str = "") -> bool:
        """Determine if we should use the larger e4b model"""
        text = (prompt + " " + context).lower()
        
        # Length-based heuristic
        if len(text) > 200:
            return True
            
        # Keyword-based complexity detection
        complex_score = sum(1 for word in self.complexity_keywords['complex'] if word in text)
        simple_score = sum(1 for word in self.complexity_keywords['simple'] if word in text)
        
        return complex_score > simple_score
    
    def get_optimal_model(self, prompt: str, context: str = "") -> str:
        """Get the optimal model considering switching costs"""
        preferred_model = "gemma3n:e4b" if self.should_use_e4b(prompt, context) else "gemma3n:e2b"
        
        # If we recently switched, stick with current model unless absolutely necessary
        time_since_switch = time.time() - self.last_switch_time
        if (self.current_model and 
            time_since_switch < self.switch_threshold and
            not self._is_switch_critical(prompt, context)):
            return self.current_model
            
        if preferred_model != self.current_model:
            self.last_switch_time = time.time()
            self.current_model = preferred_model
            
        return preferred_model
    
    def _is_switch_critical(self, prompt: str, context: str) -> bool:
        """Determine if model switch is critical despite recent switch"""
        text = (prompt + " " + context).lower()
        critical_keywords = ['urgent', 'important', 'critical', 'emergency']
        return any(word in text for word in critical_keywords)