# Gemma Fine-tuning System

This system allows you to fine-tune Gemma models using your conversation data with feedback ratings and create new Ollama model versions.

## Features

- 🔥 **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- 📊 **Feedback-based Training**: Uses `feedback_helpful` ratings to improve model quality
- 🦙 **Ollama Integration**: Creates new model versions for Ollama (doesn't overwrite existing)
- ⚙️ **Configurable**: Multiple preset configurations for different use cases
- 📈 **Validation**: Train/validation splits with early stopping
- 🔧 **Advanced Options**: Support for both Gemma-2B and Gemma-7B models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your data should be in `data/training_data.json` with this format:

```json
{
    "conversation_1": {
        "feedback_helpful": "True",
        "full_text": "Speaker_A: can you help me\nGemma: Yes, I can definitely help!...",
        "session_id": "session_20250805_115307_390",
        "timestamp": "2025-08-05 11:53:07.390773",
        "message_count": 5
    }
}
```

### 3. Run Fine-tuning

**Quick test (1 epoch, small model):**
```bash
python run_fine_tuning.py --preset quick_test
```

**Production training (5 epochs, larger model):**
```bash
python run_fine_tuning.py --preset production --model-name my-custom-gemma
```

**Default training:**
```bash
python run_fine_tuning.py
```

## Configuration Presets

- **`default`**: Balanced settings for most use cases
- **`quick_test`**: Fast training for testing (1 epoch, Gemma-2B)
- **`production`**: High-quality training (5 epochs, Gemma-7B, advanced LoRA)
- **`experimental`**: Includes negative examples and advanced filtering

## Advanced Usage

### Custom Configuration

You can modify `config.py` to create custom configurations:

```python
custom_config = FineTuningConfig(
    base_model="google/gemma-7b-it",
    num_epochs=4,
    batch_size=2,
    learning_rate=3e-5,
    use_lora=True,
    lora_r=32,
    ollama_model_name="my-expert-gemma"
)
```

### Direct Script Usage

```bash
# Basic fine-tuning
python advanced_fine_tuner.py --config production --name my-model

# Custom data path
python advanced_fine_tuner.py --data /path/to/my/data.json --config quick_test

# Skip Ollama model creation
python advanced_fine_tuner.py --no-ollama
```

### Simple Fine-tuner

For basic use cases, use the simple fine-tuner:

```bash
python gemma_fine_tuner.py --data data/training_data.json --name my-basic-model
```

## Data Filtering Options

The system can filter conversations based on:

- **Message count**: Min/max number of messages per conversation
- **Feedback quality**: Only use helpful conversations (`feedback_helpful: True`)
- **Content length**: Remove empty or too short conversations
- **Negative examples**: Optionally include unhelpful conversations for contrast learning

Configure these in `config.py`:

```python
config = FineTuningConfig(
    min_message_count=3,
    max_message_count=50,
    filter_by_feedback=True,
    include_negative_examples=False
)
```

## Output

After fine-tuning, you'll get:

1. **Fine-tuned model**: Saved in `models/gemma-finetuned-YYYYMMDD_HHMMSS/`
2. **Ollama model**: Available via `ollama run model-name`
3. **Training logs**: Detailed logs of the training process
4. **Training info**: JSON file with training metadata

## Ollama Integration

The system automatically creates Ollama-compatible models:

```bash
# After fine-tuning completes:
ollama run my-custom-gemma

# List your models:
ollama list

# Remove old versions if needed:
ollama rm old-model-name
```

## Hardware Requirements

- **GPU**: Recommended for faster training (CUDA-compatible)
- **RAM**: 16GB+ recommended for Gemma-7B, 8GB+ for Gemma-2B
- **Storage**: 10-20GB free space for model weights

## Tips for Better Results

1. **Quality Data**: Use conversations with clear feedback ratings
2. **Balanced Dataset**: Include both helpful and unhelpful examples
3. **Sufficient Data**: At least 100+ conversations for meaningful fine-tuning
4. **Validation**: Use validation split to monitor overfitting
5. **LoRA Settings**: Higher `lora_r` values for more complex adaptations

## Troubleshooting

**Out of Memory**: Reduce `batch_size` or `max_length` in config
**Slow Training**: Enable GPU support and use LoRA
**Poor Results**: Increase training data quality and quantity
**Ollama Errors**: Ensure Ollama is installed and running

## File Structure

```
fine_tuning/
├── data/
│   └── training_data.json          # Your conversation data
├── models/                         # Output directory for fine-tuned models
├── config.py                       # Configuration settings
├── gemma_fine_tuner.py            # Simple fine-tuning script
├── advanced_fine_tuner.py         # Advanced fine-tuning with LoRA
├── run_fine_tuning.py             # Easy-to-use runner script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Next Steps

1. Fine-tune your first model with the quick_test preset
2. Evaluate the results with test conversations
3. Adjust configuration based on results
4. Run production training for your final model
5. Deploy and use your custom Gemma model!