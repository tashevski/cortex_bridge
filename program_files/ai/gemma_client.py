#!/usr/bin/env python3
"""Gemma API client for simplified interactions"""

import requests
from typing import Optional

class GemmaClient:
    """Simple client for Gemma API interactions"""
    
    def __init__(self, model: str = "gemma3n:e4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate_response(self, prompt: str, context: str = "", timeout: int = 30) -> Optional[str]:
        """Generate response from Gemma"""
        full_prompt = f"{context}\nUser: {prompt}\n\nAssistant:" if context else prompt
        
        response = requests.post(
            self.api_url,
            json={'model': self.model, 'prompt': full_prompt.strip(), 'stream': False},
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        
        print(f"❌ Error: HTTP {response.status_code}")
        return None
    
    def is_server_available(self) -> bool:
        """Check if Gemma server is available"""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return response.status_code == 200 