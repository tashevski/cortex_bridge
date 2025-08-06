#!/usr/bin/env python3
"""Model preloading and warming strategies"""

import requests
import threading
import time
from typing import List, Optional
from program_files.config.config import ModelPreloaderConfig

class ModelPreloader:
    """Preload and warm models to minimize loading times"""
    
    def __init__(self, config: Optional[ModelPreloaderConfig] = None):
        if config is None:
            from config.config import cfg
            config = cfg.model_preloader
            
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.api_url = f"{config.base_url}/api/generate"
    
    def warm_model(self, model: str) -> float:
        """Warm up a model with a minimal request"""
        start_time = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    'model': model, 
                    'prompt': 'Hi', 
                    'stream': False,
                    'options': {'num_predict': 1}  # Minimal generation
                },
                timeout=self.timeout
            )
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ {model} warmed up in {load_time:.2f}s")
                return load_time
            else:
                print(f"❌ Failed to warm {model}: HTTP {response.status_code}")
                return float('inf')
                
        except Exception as e:
            print(f"❌ Error warming {model}: {e}")
            return float('inf')
    
    def preload_models_parallel(self, models: List[str]):
        """Preload multiple models in parallel (for systems with enough VRAM)"""
        threads = []
        
        for model in models:
            thread = threading.Thread(target=self.warm_model, args=(model,))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
    
    def background_model_rotation(self, models: List[str], interval: int = 300):
        """Rotate models in background to keep them warm"""
        def rotate():
            while True:
                for model in models:
                    self.warm_model(model)
                    time.sleep(interval)
        
        thread = threading.Thread(target=rotate, daemon=True)
        thread.start()
        print(f"🔄 Background model rotation started (every {interval}s)")
    
    def get_model_load_times(self, models: List[str]) -> dict:
        """Benchmark model loading times"""
        load_times = {}
        
        # Unload all models first
        for model in models:
            requests.post(f"{self.base_url}/api/generate", 
                         json={'model': model, 'keep_alive': 0})
        
        time.sleep(2)  # Wait for unload
        
        # Time each model load
        for model in models:
            load_times[model] = self.warm_model(model)
            
        return load_times