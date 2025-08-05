"""Minimal configuration for Gemma fine-tuning"""

# Quick presets
PRESETS = {
    "test": {"model": "google/gemma-2b-it", "epochs": 1, "batch": 1},
    "prod": {"model": "google/gemma-7b-it", "epochs": 3, "batch": 2}, 
    "fast": {"model": "google/gemma-2b-it", "epochs": 2, "batch": 4}
}