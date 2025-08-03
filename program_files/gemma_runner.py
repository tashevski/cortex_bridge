#!/usr/bin/env python3
"""Simple Gemma 3N runner with Ollama"""

import subprocess
import time
import requests
import signal
import sys
import json
import base64
import os

class GemmaRunner:
    def __init__(self, model="gemma3n:e2b"):
        self.model = model
        self.server = None
    
    def start_server(self):
        """Start Ollama server"""
        print("Starting Ollama...")
        self.server = subprocess.Popen(['ollama', 'serve'])
        time.sleep(3)  # Wait for server
    
    def pull_model(self):
        """Pull the model if needed"""
        print(f"Pulling {self.model}...")
        subprocess.run(['ollama', 'pull', self.model], check=True)
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def chat(self, image_path=None, prompt_template=None):
        """Interactive chat loop with optional image and template"""
        print(f" {self.model} ready! Type 'quit' to exit\n")
        
        # Load image if provided
        image_data = None
        if image_path and os.path.exists(image_path):
            image_data = self.encode_image(image_path)
            if image_data:
                print(f"📷 Loaded image: {image_path}")
        
        # Load prompt template if provided
        template = None
        if prompt_template and os.path.exists(prompt_template):
            try:
                with open(prompt_template, 'r') as f:
                    template = f.read().strip()
                print(f"�� Loaded template: {prompt_template}")
            except Exception as e:
                print(f"Error loading template: {e}")
        
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Apply template if available
                if template:
                    final_prompt = template.format(prompt=prompt)
                else:
                    final_prompt = prompt
                
                # Prepare request payload
                payload = {
                    'model': self.model,
                    'prompt': final_prompt,
                    'stream': False
                }
                
                # Add image if available
                if image_data:
                    payload['images'] = [image_data]
                
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=payload
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"Gemma: {data['response']}\n")
                    except json.JSONDecodeError:
                        print(f"Error parsing response\n")
                    
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Stop server"""
        if self.server:
            self.server.terminate()
            print("✅ Stopped Ollama")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemma 3N Chat Runner')
    parser.add_argument('--image', '-i', help='Path to image file')
    parser.add_argument('--template', '-t', help='Path to prompt template file')
    parser.add_argument('--model', '-m', default='gemma3n:e2b', help='Model name')
    
    args = parser.parse_args()
    
    runner = GemmaRunner(args.model)
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        runner.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        runner.start_server()
        runner.pull_model()
        runner.chat(image_path=args.image, prompt_template=args.template)
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()