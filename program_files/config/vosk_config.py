#!/usr/bin/env python3
"""Vosk model configuration"""

import os
from pathlib import Path
from typing import Dict

MODELS = {
    "small": {"name": "vosk-model-small-en-us-0.15", "accuracy": "basic"},
    "medium": {"name": "vosk-model-en-us-0.22", "accuracy": "good"}
}

def get_vosk_model_path() -> str:
    """Get the path to the current Vosk model"""
    base_dir = Path(__file__).parent.parent
    
    # Try medium first, fallback to small
    for model_type in ["medium", "small"]:
        model_path = base_dir / "models" / MODELS[model_type]["name"]
        if model_path.exists():
            return str(model_path)
    
    # Default fallback
    return str(base_dir / "models" / MODELS["medium"]["name"])

def get_vosk_model_info() -> Dict:
    """Get information about the current Vosk model"""
    model_path = get_vosk_model_path()
    model_name = Path(model_path).name
    
    for model_type, info in MODELS.items():
        if info["name"] == model_name:
            return {"name": model_name, "accuracy": info["accuracy"]}
    
    return {"name": model_name, "accuracy": "unknown"} 