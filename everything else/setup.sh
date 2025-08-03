#!/bin/bash
# Setup script for Real-Time Audio Transcription System

echo "🚀 Setting up Real-Time Audio Transcription System"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip found"

# Detect OS and install system dependencies
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

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Test the installation
echo "🧪 Testing installation..."
python3 test_audio.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage:"
echo "  python3 audio_transcriber.py    # Basic audio transcription"
echo "  python3 voice_gemma.py          # Voice chat with Gemma (requires Ollama)"
echo "  python3 gemma_runner.py         # Text chat with Gemma"
echo ""
echo "For voice chat with Gemma, install Ollama from: https://ollama.ai/" 