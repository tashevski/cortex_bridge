#!/usr/bin/env python3
"""Smart model selection to minimize loading overhead"""

import time
from typing import Optional
from config.config import SmartModelSelectorConfig

class SmartModelSelector:
    """Intelligently selects models to minimize loading overhead"""
    
    def __init__(self, config: Optional[SmartModelSelectorConfig] = None):
        if config is None:
            from config.config import cfg
            config = cfg.smart_model_selector
            
        self.current_model = None
        self.last_switch_time = 0
        self.switch_threshold = config.switch_threshold
        self.context_length_threshold = config.context_length_threshold
        
        self.complexity_keywords = {
            'complex': config.complex_keywords,
            'simple': config.simple_keywords
        }
    
    def should_use_e4b(self, prompt: str, context: str = "", has_image: bool = False) -> bool:
        """Determine if we should use the larger e4b model"""
        # Always use e4b for image inputs (multimodal requires more capability)
        if has_image:
            return True
            
        text = (prompt + " " + context).lower()
        
        # Context length-based heuristic (configurable threshold)
        if len(context) > self.context_length_threshold:
            return True
            
        # Keyword-based complexity detection
        complex_score = sum(1 for word in self.complexity_keywords['complex'] if word in text)
        simple_score = sum(1 for word in self.complexity_keywords['simple'] if word in text)
        
        return complex_score > simple_score
    
    def get_optimal_model(self, prompt: str, context: str = "", has_image: bool = False) -> str:
        """Get the optimal model considering switching costs"""
        preferred_model = "gemma3n:e4b" if self.should_use_e4b(prompt, context, has_image) else "gemma3n:e2b"
        
        # If we recently switched, stick with current model
        time_since_switch = time.time() - self.last_switch_time
        if self.current_model and time_since_switch < self.switch_threshold:
            return self.current_model
            
        if preferred_model != self.current_model:
            self.last_switch_time = time.time()
            self.current_model = preferred_model
            
        return preferred_model
    
