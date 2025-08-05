#!/bin/bash

echo "🚀 Setting up Gemma Fine-tuning Environment"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed. Please install Ollama from https://ollama.ai"
    echo "   This is required for creating Ollama-compatible models."
else
    echo "✅ Ollama is available"
fi

# Check GPU availability
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
else:
    print('ℹ️  No GPU detected - fine-tuning will use CPU (slower)')
"

# Test imports
echo "🧪 Testing imports..."
python3 -c "
try:
    import torch
    import transformers
    import datasets
    import peft
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Quick start:"
echo "  python3 run_fine_tuning.py --preset quick_test"
echo ""
echo "For more options, see README.md"