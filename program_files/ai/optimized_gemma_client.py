#!/usr/bin/env python3
"""Optimized GemmaClient with smart loading strategies"""

from .gemma_client import GemmaClient
from .smart_model_selector import SmartModelSelector  
from .model_preloader import ModelPreloader
from .latency_monitor import LatencyMonitor
from utils.config import GemmaClientConfig
import requests
import time
from typing import Optional

class OptimizedGemmaClient(GemmaClient):
    """Enhanced GemmaClient with loading optimizations"""
    
    def __init__(self, config: Optional[GemmaClientConfig] = None):
        if config is None:
            from utils.config import cfg
            config = cfg.gemma_client
            
        super().__init__(config.default_model, config.base_url)
        self.selector = SmartModelSelector()  # Uses default config
        self.preloader = ModelPreloader()  # Uses default config
        self.latency_monitor = LatencyMonitor()  # Uses default config
        self.current_loaded_model = None
        
    def generate_response_optimized(self, prompt: str, context: str = "", **kwargs):
        """Generate response with optimized model selection and latency monitoring"""
        
        # Check if image is provided
        has_image = 'image_path' in kwargs and kwargs['image_path'] is not None
        
        # Get optimal model from selector
        optimal_model = self.selector.get_optimal_model(prompt, context, has_image)
        
        # Apply latency-based adjustments
        final_model, reason = self.latency_monitor.get_model_recommendation(optimal_model)
        
        if reason.startswith("🚨"):
            print(reason)
        
        # Check if we need to switch models
        if final_model != self.current_loaded_model:
            print(f"🔄 Switching to {final_model}...")
            
            # Unload current model to free VRAM
            if self.current_loaded_model:
                self._unload_model(self.current_loaded_model)
            
            # Warm up new model
            load_time = self.preloader.warm_model(final_model)
            print(f"⚡ Model loaded in {load_time:.2f}s")
            
            self.current_loaded_model = final_model
            self.model = final_model
        
        # Start latency monitoring
        self.latency_monitor.start_response_timing(
            model=final_model,
            context_length=len(context),
            has_image=has_image
        )
        
        model_switched = final_model != self.current_loaded_model
        switch_reason = reason if model_switched else ""
        
        try:
            # Generate response
            response = self.generate_response(prompt, context, **kwargs)
            return response
        finally:
            # End latency monitoring
            metrics = self.latency_monitor.end_response_timing()
            if metrics:
                # Store metrics for database
                self._last_latency_metrics = {
                    'response_time': metrics.response_time,
                    'user_spoke_during_response': metrics.user_spoke_during_response,
                    'speech_activity_during_response': metrics.speech_activity_during_response,
                    'model_used': metrics.model_used,
                    'context_length': metrics.context_length,
                    'had_image': metrics.had_image,
                    'model_switched': model_switched,
                    'switch_reason': switch_reason
                }
                
                if metrics.response_time > 3.0:
                    print(f"⚠️  Slow response: {metrics.response_time:.2f}s")
                if metrics.user_spoke_during_response:
                    print(f"🗣️  User spoke for {metrics.speech_activity_during_response:.1f}s during response")
                if model_switched:
                    print(f"🔄 Model switched: {switch_reason}")
    
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
    
    def record_speech_activity(self, is_speech: bool):
        """Record user speech activity for latency monitoring"""
        self.latency_monitor.record_speech_activity(is_speech)
    
    def get_latency_status(self):
        """Get current latency monitoring status"""
        return self.latency_monitor.get_latency_analysis()
    
    def print_latency_status(self):
        """Print latency monitoring status"""
        self.latency_monitor.print_status()
    
    def get_last_latency_metrics(self) -> Optional[dict]:
        """Get the latency metrics from the last response"""
        return getattr(self, '_last_latency_metrics', None)