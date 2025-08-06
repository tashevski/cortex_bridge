#!/usr/bin/env python3
"""Simple TTS Test Script"""

import os
import sys
import time

# Add program_files to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'program_files'))

from tts.tts_personal import OfflineTTSFile

def TTS(text):
    """Simple TTS function"""
    tts.stream_text_to_speech(text)

def pause(seconds):
    """Pause for specified seconds"""
    print(f"Pausing {seconds} seconds...")
    time.sleep(seconds)

def big_number(num):
    """Print big ASCII art numbers"""
    digits = {
        '0': ["  ███  ", " █   █ ", " █   █ ", " █   █ ", "  ███  "],
        '1': ["   █   ", "  ██   ", "   █   ", "   █   ", " █████ "],
        '2': [" █████ ", "     █ ", " █████ ", " █     ", " █████ "],
        '3': [" █████ ", "     █ ", " █████ ", "     █ ", " █████ "],
        '4': [" █   █ ", " █   █ ", " █████ ", "     █ ", "     █ "],
        '5': [" █████ ", " █     ", " █████ ", "     █ ", " █████ "],
        '6': [" █████ ", " █     ", " █████ ", " █   █ ", " █████ "],
        '7': [" █████ ", "     █ ", "    █  ", "   █   ", "  █    "],
        '8': [" █████ ", " █   █ ", " █████ ", " █   █ ", " █████ "],
        '9': [" █████ ", " █   █ ", " █████ ", "     █ ", " █████ "]
    }
    
    lines = [""] * 5
    for digit in str(num):
        if digit in digits:
            for i in range(5):
                lines[i] += digits[digit][i] + "  "
    
    print("\n" + "="*60)
    for line in lines:
        print(f"    {line}")
    print("="*60 + "\n")

def pause_timer(seconds):
    """Pause for specified seconds with countdown"""
    for i in range(seconds, 0, -1):
        big_number(i)
        time.sleep(1)
    
    print("\n" + "="*60)
    print("                    RESUMING...")
    print("="*60 + "\n")

# Initialize TTS
tts = OfflineTTSFile()

pause_timer(10)

# Test sequence
TTS("Hello how are you?")
pause(10)

TTS("This is the second test message")
pause(5)

TTS("And this is the final test")
pause(3)

print("Test complete!")