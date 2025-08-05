# Gemma Fine-tuning System

A clean, organized system for fine-tuning Gemma models with LoRA and Ollama integration.

## 📁 Project Structure

```
fine_tuning/
├── src/                          # Core fine-tuning modules
│   ├── advanced_fine_tuner.py    # Full-featured fine-tuner with LoRA
│   ├── gemma_fine_tuner.py       # Basic fine-tuning implementation  
│   └── minimal_fine_tuner.py     # Lightweight fine-tuner
├── config/                       # Configuration files
│   ├── config.py                 # Main configuration with presets
│   └── config_minimal.py         # Minimal configuration options
├── utils/                        # Utility scripts
│   ├── evaluate_model.py         # Model evaluation and comparison
│   └── setup.sh                  # Environment setup script
├── docs/                         # Documentation
│   ├── README.md                 # Detailed documentation
│   └── README_minimal.md         # Quick start guide
├── examples/                     # Example usage scripts
│   └── run.py                    # Simple example runner
├── data/                         # Training data
│   ├── training_data.json        # Your conversation data
│   └── example_data.json         # Example data format
├── run_fine_tuning.py           # Main entry point
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
# or
bash utils/setup.sh
```

### 2. Run Fine-tuning
```bash
# Quick test (1 epoch, small model)
python run_fine_tuning.py --preset quick_test

# Production training (5 epochs, larger model)
python run_fine_tuning.py --preset production --model-name my-custom-gemma

# Default training
python run_fine_tuning.py
```

### 3. Evaluate Model
```bash
python utils/evaluate_model.py my-custom-gemma gemma:2b
```

## 🎯 Key Features

- **📦 Organized Structure**: Clean separation of core, config, utils, and docs
- **🔥 LoRA Fine-tuning**: Efficient training using Low-Rank Adaptation
- **📊 Feedback-based**: Uses `feedback_helpful` ratings for quality training
- **🦙 Ollama Integration**: Creates versioned models without overwriting
- **⚙️ Multiple Presets**: Quick test, production, and experimental configs
- **🧪 Model Evaluation**: Built-in testing and comparison tools

## 📋 Configuration Presets

- **`default`**: Balanced settings (3 epochs, Gemma-2B, LoRA)
- **`quick_test`**: Fast training (1 epoch, smaller batches)
- **`production`**: High-quality (5 epochs, Gemma-7B, advanced LoRA)
- **`experimental`**: Includes negative examples and advanced filtering

## 💻 Advanced Usage

### Custom Configuration
```python
from config.config import FineTuningConfig

custom_config = FineTuningConfig(
    base_model="google/gemma-7b-it",
    num_epochs=4,
    batch_size=2,
    learning_rate=3e-5,
    use_lora=True,
    lora_r=32
)
```

### Direct Module Usage
```python
from src.advanced_fine_tuner import AdvancedGemmaFineTuner
from config.config import get_config

config = get_config("production")
tuner = AdvancedGemmaFineTuner(config)
model_path, ollama_name = tuner.run_fine_tuning(model_name="my-model")
```

## 📖 Documentation

- **[docs/README.md](docs/README.md)**: Complete feature documentation
- **[docs/README_minimal.md](docs/README_minimal.md)**: Quick start guide
- **[examples/](examples/)**: Usage examples and sample scripts

## 🔧 Data Format

Your data should be in `data/training_data.json`:

```json
{
  "conversation_1": {
    "feedback_helpful": "True",
    "full_text": "Speaker_A: question\nGemma: answer\n...",
    "session_id": "session_id",
    "timestamp": "timestamp",
    "message_count": 5
  }
}
```

## 🎉 Output

After training:
- Fine-tuned model in `models/gemma-finetuned-YYYYMMDD_HHMMSS/`
- Ollama model available via `ollama run model-name`
- Training logs and metadata included

## 🤝 Contributing

The modular structure makes it easy to:
- Add new fine-tuning strategies in `src/`
- Create new configuration presets in `config/`
- Add evaluation metrics in `utils/`
- Include usage examples in `examples/`