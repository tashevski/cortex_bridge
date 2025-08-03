#!/usr/bin/env python3
"""Minimal offline transcription"""

import pyaudio
import json
from vosk import Model, KaldiRecognizer

# Download Vosk model if not available (50MB English model)
try:
    model = Model("vosk-model-small-en-us-0.15")
except:
    print("Downloading model...")
    import urllib.request, zipfile, os
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    urllib.request.urlretrieve(url, "model.zip")
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("model.zip")
    model = Model("vosk-model-small-en-us-0.15")

# Initialize speech recognizer with 16kHz sample rate
rec = KaldiRecognizer(model, 16000)
audio = pyaudio.PyAudio()

# Open audio stream from microphone
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                   input=True, frames_per_buffer=1024)
stream.start_stream()

print("🎤 Speak (OFFLINE)... (Ctrl+C to stop)")

try:
    while True:
        # Read audio data in chunks
        data = stream.read(1024)
        
        # Process audio through Vosk recognizer
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get('text', '').strip():
                print(f"📝 {result['text']}")
except KeyboardInterrupt:
    # Clean shutdown
    stream.stop_stream()
    stream.close()
    audio.terminate() 