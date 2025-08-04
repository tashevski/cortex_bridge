#!/usr/bin/env python3
"""Optimized GemmaClient with smart loading strategies"""

from .gemma_client import GemmaClient
from .smart_model_selector import SmartModelSelector  
from .model_preloader import ModelPreloader
import requests
import time

class OptimizedGemmaClient(GemmaClient):
    """Enhanced GemmaClient with loading optimizations"""
    
    def __init__(self, default_model: str = "gemma3n:e2b", base_url: str = "http://localhost:11434",
                 context_length_threshold: int = 500,
                 complex_keywords: list = None,
                 simple_keywords: list = None):
        super().__init__(default_model, base_url)
        self.selector = SmartModelSelector(
            context_length_threshold=context_length_threshold,
            complex_keywords=complex_keywords,
            simple_keywords=simple_keywords
        )
        self.preloader = ModelPreloader(base_url)
        self.current_loaded_model = None
        
    def generate_response_optimized(self, prompt: str, context: str = "", **kwargs):
        """Generate response with optimized model selection"""
        
        # Check if image is provided
        has_image = 'image_path' in kwargs and kwargs['image_path'] is not None
        
        # Get optimal model
        optimal_model = self.selector.get_optimal_model(prompt, context, has_image)
        
        # Check if we need to switch models
        if optimal_model != self.current_loaded_model:
            print(f"🔄 Switching to {optimal_model}...")
            
            # Unload current model to free VRAM
            if self.current_loaded_model:
                self._unload_model(self.current_loaded_model)
            
            # Warm up new model
            load_time = self.preloader.warm_model(optimal_model)
            print(f"⚡ Model loaded in {load_time:.2f}s")
            
            self.current_loaded_model = optimal_model
            self.model = optimal_model
        
        # Generate response
        return self.generate_response(prompt, context, **kwargs)
    
    def _unload_model(self, model: str):
        """Explicitly unload a model to free VRAM"""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={'model': model, 'keep_alive': 0},
                timeout=10
            )
            print(f"🗑️  Unloaded {model}")
        except Exception as e:
            print(f"⚠️  Failed to unload {model}: {e}")
    
    def benchmark_switching(self):
        """Benchmark model switching performance"""
        models = ["gemma3n:e2b", "gemma3n:e4b"]
        results = {}
        
        for model in models:
            print(f"\n🧪 Testing {model} loading time...")
            
            # Unload all models
            for m in models:
                self._unload_model(m)
            time.sleep(2)
            
            # Time the load
            start = time.time()
            self.preloader.warm_model(model)
            load_time = time.time() - start
            
            results[model] = load_time
            print(f"   ⏱️  {model}: {load_time:.2f}s")
        
        return results