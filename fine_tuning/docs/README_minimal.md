# Minimal Gemma Fine-tuning

Ultra-clean fine-tuning for Gemma models with LoRA and Ollama integration.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Test run (fast)
python run.py test

# Production run
python run.py prod my-custom-gemma

# Direct usage
python minimal_fine_tuner.py --data data/training_data.json --name my-model
```

## Files

- `minimal_fine_tuner.py` - Core fine-tuning (90 lines)
- `run.py` - Simple runner with presets (25 lines)  
- `config_minimal.py` - Basic presets (5 lines)

## Features

- ✅ LoRA fine-tuning for efficiency
- ✅ Auto-filters helpful conversations (`feedback_helpful: true`)
- ✅ Creates versioned Ollama models
- ✅ GPU/CPU auto-detection
- ✅ Minimal dependencies

## Usage

Your data in `data/training_data.json` should have:
```json
{
  "conversation_1": {
    "feedback_helpful": "True",
    "full_text": "Speaker_A: question\nGemma: answer\n..."
  }
}
```

Output: `ollama run gemma-ft-YYYYMMDD_HHMMSS`