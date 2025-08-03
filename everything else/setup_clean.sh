#!/bin/bash
# Setup script for Clean Environment with ML-based Emotion Classification

echo "🚀 Setting up Clean Environment for ML-based Emotion Classification"
echo "================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✅ Conda found"

# Create clean environment
echo "🔧 Creating clean environment 'emotion_env'..."
conda create -n emotion_env python=3.11 -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emotion_env

# Install PyTorch (CPU version for compatibility)
echo "📦 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other ML libraries
echo "📦 Installing ML libraries..."
pip install transformers sentence-transformers

# Install audio and other dependencies
echo "📦 Installing audio dependencies..."
pip install pyaudio vosk textblob

# Install system dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "✅ Homebrew found"
    
    # Install portaudio
    echo "📦 Installing portaudio..."
    brew install portaudio
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing dependencies with apt-get..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    elif command -v yum &> /dev/null; then
        echo "📦 Installing dependencies with yum..."
        sudo yum install -y portaudio-devel python3-pyaudio
    elif command -v pacman &> /dev/null; then
        echo "📦 Installing dependencies with pacman..."
        sudo pacman -S portaudio python-pyaudio
    else
        echo "⚠️  Could not detect package manager. Please install portaudio manually."
    fi
else
    echo "⚠️  Unsupported OS. Please install portaudio manually."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use the system:"
echo "  1. Activate the environment: conda activate emotion_env"
echo "  2. Run transcription: python transcriber.py"
echo "  3. Run voice chat: python simple_voice_gemma.py"
echo ""
echo "The system now uses proper ML-based emotion classification!" 