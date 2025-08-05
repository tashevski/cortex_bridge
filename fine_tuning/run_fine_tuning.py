#!/usr/bin/env python3
"""Simple fine-tuning runner with presets"""

import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.advanced_fine_tuner import AdvancedGemmaFineTuner
from config.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Run Gemma fine-tuning")
    parser.add_argument("--preset", default="default", 
                       choices=["default", "quick_test", "production", "experimental"])
    parser.add_argument("--model-name", help="Custom Ollama model name")    
    args = parser.parse_args()
    
    config = get_config(args.preset)
    print(f"🚀 Fine-tuning with {args.preset} preset")
    print(f"📊 Model: {config.base_model} | Epochs: {config.num_epochs} | Batch: {config.batch_size}")
    
    fine_tuner = AdvancedGemmaFineTuner(config)
    model_path, ollama_name = fine_tuner.run_fine_tuning(model_name=args.model_name)
    
    print(f"\n🎉 Success!\n📁 Model: {model_path}")
    if ollama_name:
        print(f"🦙 Ollama: {ollama_name}\n💻 Usage: ollama run {ollama_name}")

if __name__ == "__main__":
    main()