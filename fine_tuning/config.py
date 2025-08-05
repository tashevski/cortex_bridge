"""
Configuration file for Gemma fine-tuning
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class FineTuningConfig:
    """Configuration class for fine-tuning parameters."""
    
    # Data settings
    data_path: str = "data/training_data.json"
    validation_split: float = 0.1
    
    # Model settings
    base_model: str = "google/gemma-2b-it"  # Can also use "google/gemma-7b-it"
    max_length: int = 512
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    
    # LoRA settings (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Output settings
    output_dir: str = "models"
    version_suffix: Optional[str] = None
    save_steps: int = 500
    logging_steps: int = 10
    
    # Ollama settings
    create_ollama_model: bool = True
    ollama_model_name: Optional[str] = None
    
    # Hardware settings
    use_gpu: bool = True
    use_fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Conversation filtering
    min_message_count: int = 3  # Minimum messages in conversation
    max_message_count: int = 50  # Maximum messages in conversation
    filter_by_feedback: bool = True  # Only use helpful conversations
    include_negative_examples: bool = False  # Include unhelpful conversations as negative examples

# Default configuration
DEFAULT_CONFIG = FineTuningConfig()

# Model configurations for different use cases
CONFIGS = {
    "quick_test": FineTuningConfig(
        num_epochs=1,
        batch_size=1,
        max_length=256,
        save_steps=100,
        base_model="google/gemma-2b-it"
    ),
    
    "production": FineTuningConfig(
        num_epochs=5,
        batch_size=4,
        max_length=1024,
        learning_rate=3e-5,
        base_model="google/gemma-7b-it",
        use_lora=True,
        lora_r=32
    ),
    
    "experimental": FineTuningConfig(
        include_negative_examples=True,
        filter_by_feedback=False,
        num_epochs=4,
        learning_rate=2e-5
    )
}

def get_config(config_name: str = "default") -> FineTuningConfig:
    """Get configuration by name."""
    if config_name == "default":
        return DEFAULT_CONFIG
    elif config_name in CONFIGS:
        return CONFIGS[config_name]
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(CONFIGS.keys())}")