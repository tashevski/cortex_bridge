# Offline Real-Time Audio Transcription System

This project provides offline real-time audio-to-text transcription capabilities with optional integration with Gemma language models for voice-enabled AI conversations.

## Features

- 🎤 **Real-time audio transcription** - Continuously capture and transcribe audio input
- 🔴 **Completely offline** - No internet connection required after setup
- 🤖 **Voice-enabled AI chat** - Integrate with Gemma models for voice conversations
- 🎯 **Smart silence detection** - Automatically trigger transcription on pauses
- 🔒 **Privacy-focused** - All processing happens locally

## Files

- `transcriber.py` - **Offline** real-time audio transcription (Vosk - no internet required)
- `simple_voice_gemma.py` - Voice-enabled chat with Gemma AI models
- `gemma_runner.py` - Original text-based Gemma interface
- `requirements.txt` - Python dependencies

## Prerequisites

1. **Python 3.7+**
2. **Ollama** (for Gemma models) - [Install Ollama](https://ollama.ai/)
3. **Microphone access** on your system
4. **Internet connection** (only for initial model download)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (macOS):**
   ```bash
   # Install portaudio (required for PyAudio)
   brew install portaudio
   ```

4. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   # Install portaudio and other audio dependencies
   sudo apt-get update
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

## Usage

### 1. Audio Transcription

**Offline (Vosk - No Internet Required):**
```bash
python transcriber.py
```

**Note:** The system will automatically download a 50MB model on first run.

### 2. Voice-Enabled Gemma Chat

For voice conversations with Gemma AI models:

```bash
python simple_voice_gemma.py
```

**Note:** This requires Ollama to be installed.

### 3. Text-Based Gemma Chat

For traditional text-based interactions:

```bash
python gemma_runner.py
```

**Options:**
- `--model gemma3n:e2b` - Model name
- `--image path/to/image.jpg` - Include image in conversation
- `--template path/to/template.txt` - Use prompt template

## How It Works

### Audio Transcription Flow

1. **Audio Capture**: Continuously captures audio from your microphone
2. **Silence Detection**: Monitors audio levels to detect speech pauses
3. **Buffer Management**: Maintains a rolling buffer of recent audio
4. **Transcription Trigger**: When silence is detected, processes the audio buffer
5. **Speech Recognition**: Uses Vosk (offline) to convert audio to text
6. **Real-time Output**: Displays transcribed text immediately

### Voice Chat Flow

1. **Audio Transcription**: Captures and transcribes your speech
2. **Text Processing**: Sends transcribed text to Gemma model
3. **AI Response**: Receives and displays Gemma's response
4. **Continuous Loop**: Repeats for natural conversation flow

## Configuration

### Audio Settings

- **Sample Rate**: Higher rates (44.1kHz) for better quality, lower rates (16kHz) for efficiency
- **Chunk Size**: Smaller chunks for lower latency, larger chunks for stability
- **Silence Threshold**: Adjust based on your microphone and environment
- **Silence Duration**: How long to wait before processing speech

### Language Support

The system currently supports English with the default Vosk model. Additional language models can be downloaded from the [Vosk model repository](https://alphacephei.com/vosk/models).

## Troubleshooting

### Common Issues

1. **"No module named 'pyaudio'"**
   - Install portaudio: `brew install portaudio` (macOS) or `sudo apt-get install portaudio19-dev` (Ubuntu)
   - Reinstall PyAudio: `pip install --force-reinstall pyaudio`

2. **"Error starting audio stream"**
   - Check microphone permissions
   - Ensure microphone is not in use by other applications
   - Try different audio devices

3. **"Speech recognition service error"**
   - Check microphone is working
   - Verify Vosk model is properly downloaded

4. **"Error starting Ollama server"**
   - Ensure Ollama is installed: https://ollama.ai/
   - Check if Ollama is already running: `ollama list`

### Performance Tips

- **Use wired microphones** for better audio quality
- **Close other audio applications** to reduce interference
- **Ensure good lighting** for better speech recognition accuracy

## Advanced Usage

### Custom Models

You can use different Gemma models by editing the model name in `simple_voice_gemma.py`:

```python
SimpleVoiceGemma(model="gemma3n:2b").start()
```

### Integration with Other Systems

The transcription system can be easily integrated with other applications by importing the classes:

```python
from transcriber import OfflineTranscriber
from simple_voice_gemma import SimpleVoiceGemma

# Use the classes in your own applications
```

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 