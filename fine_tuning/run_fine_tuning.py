#!/usr/bin/env python3
"""
Simple script to run fine-tuning with different configurations
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from advanced_fine_tuner import AdvancedGemmaFineTuner
from config import get_config

def main():
    parser = argparse.ArgumentParser(description="Run Gemma fine-tuning")
    parser.add_argument("--preset", default="default", 
                       choices=["default", "quick_test", "production", "experimental"],
                       help="Configuration preset")
    parser.add_argument("--model-name", help="Custom Ollama model name")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting fine-tuning with preset: {args.preset}")
    
    # Get configuration
    config = get_config(args.preset)
    
    # Print configuration summary
    print(f"📊 Configuration Summary:")
    print(f"   Base model: {config.base_model}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Use LoRA: {config.use_lora}")
    print(f"   Max length: {config.max_length}")
    print()
    
    # Create and run fine-tuner
    fine_tuner = AdvancedGemmaFineTuner(config)
    
    try:
        model_path, ollama_name = fine_tuner.run_fine_tuning(model_name=args.model_name)
        
        print(f"\n🎉 Success!")
        print(f"📁 Model: {model_path}")
        if ollama_name:
            print(f"🦙 Ollama: {ollama_name}")
            print(f"💻 Usage: ollama run {ollama_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()