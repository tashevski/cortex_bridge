#!/usr/bin/env python3
"""Minimal voice chat with Gemma"""

import pyaudio
import speech_recognition as sr
import numpy as np
import threading
import queue
import requests
import subprocess
import time

class SimpleVoiceGemma:
    def __init__(self, model="gemma3n:e2b"):
        self.model = model
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.running = False
        
    def start(self):
        # Start Ollama server and pull model
        subprocess.Popen(['ollama', 'serve'])
        time.sleep(2)
        subprocess.run(['ollama', 'pull', self.model])
        
        self.running = True
        # Open audio stream for microphone input
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=16000,
            input=True, frames_per_buffer=1024,
            stream_callback=self._callback
        )
        
        # Start processing thread
        threading.Thread(target=self._process, daemon=True).start()
        self.stream.start_stream()
        
        print("🎤 Speak to Gemma... (Ctrl+C to stop)")
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def _callback(self, in_data, frame_count, time_info, status):
        # Audio callback - detect when there's sound
        if self.running:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            if np.abs(audio_data).mean() > 1000:  # Noise threshold
                self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process(self):
        buffer = []
        while self.running:
            try:
                data = self.audio_queue.get(timeout=1)
                buffer.extend(data)
                
                # Process 2 seconds of audio at a time
                if len(buffer) > 16000 * 2:
                    try:
                        # Convert audio to text using Google Speech Recognition
                        audio = sr.AudioData(
                            np.array(buffer, dtype=np.int16).tobytes(),
                            16000, 2
                        )
                        text = self.recognizer.recognize_google(audio)
                        if text:
                            print(f"🎤 You: {text}")
                            
                            # Send text to Gemma and get response
                            response = requests.post(
                                'http://localhost:11434/api/generate',
                                json={'model': self.model, 'prompt': text, 'stream': False}
                            ).json()
                            
                            print(f"🤖 Gemma: {response['response']}\n")
                    except:
                        pass
                    buffer = []
            except queue.Empty:
                pass
    
    def stop(self):
        # Clean shutdown
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    SimpleVoiceGemma().start() 