#!/usr/bin/env python3
"""Minimal runner for Gemma fine-tuning"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.minimal_fine_tuner import MinimalGemmaFineTuner
from config.config_minimal import PRESETS

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <preset> [model_name]")
        print(f"Presets: {list(PRESETS.keys())}")
        return
    
    preset_name = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if preset_name not in PRESETS:
        print(f"Unknown preset. Available: {list(PRESETS.keys())}")
        return
    
    preset = PRESETS[preset_name]
    tuner = MinimalGemmaFineTuner(model=preset["model"])
    
    # Override training args if needed
    if preset_name == "test":
        tuner.epochs = 1
        tuner.batch_size = 1
    
    result = tuner.run(model_name)
    if result:
        print(f"✅ Done! Use: ollama run {result}")

if __name__ == "__main__":
    main()