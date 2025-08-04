#!/usr/bin/env python3
"""Gemma API client for simplified interactions"""

import requests
import base64
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path

class GemmaClient:
    """Simple client for Gemma API interactions"""
    
    def __init__(self, model: str = "gemma3n:e4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API transmission"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _format_prompt_with_template(self, prompt: str, context: str = "", 
                                   prompt_template: Optional[str] = None) -> str:
        """Format prompt using template or default formatting"""
        if prompt_template:
            return prompt_template.format(context=context, prompt=prompt)
        return f"{context}\nUser: {prompt}\n\nAssistant:" if context else prompt
    
    def generate_response(self, prompt: str, context: str = "", timeout: Optional[int] = None, 
                         image_path: Optional[Union[str, Path]] = None,
                         prompt_template: Optional[str] = None,
                         vector_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate response from Gemma with enhanced input options
        
        Args:
            prompt: The main prompt text
            context: Additional context text
            timeout: Request timeout in seconds
            image_path: Path to image file for multimodal input
            prompt_template: Template string with {context} and {prompt} placeholders
            vector_context: JSON object containing vector database context or metadata
        """
        # Format the prompt using template if provided
        full_prompt = self._format_prompt_with_template(prompt, context, prompt_template)
        
        # Add vector context if provided
        if vector_context:
            vector_context_str = f"Vector context: {json.dumps(vector_context, indent=2)}\n\n"
            full_prompt = vector_context_str + full_prompt
        
        # Prepare request payload
        payload = {
            'model': self.model, 
            'prompt': full_prompt.strip(), 
            'stream': False
        }
        
        # Add image if provided
        if image_path:
            try:
                encoded_image = self._encode_image(image_path)
                payload['images'] = [encoded_image]
            except Exception as e:
                print(f"❌ Error encoding image: {e}")
                return None
        
        # Use default timeout if none provided
        request_timeout = timeout if timeout is not None else 30
        response = requests.post(self.api_url, json=payload, timeout=request_timeout)
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        
        print(f"❌ Error: HTTP {response.status_code}")
        return None
    
    def is_server_available(self) -> bool:
        """Check if Gemma server is available"""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return response.status_code == 200 