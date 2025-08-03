# Offline Real-Time Audio Transcription System

This project provides offline real-time audio-to-text transcription capabilities with optional integration with Gemma language models for voice-enabled AI conversations.

## Features

- 🎤 **Real-time audio transcription** - Continuously capture and transcribe audio input
- 🔴 **Completely offline** - No internet connection required after setup
- 🤖 **Voice-enabled AI chat** - Integrate with Gemma models for voice conversations
- 🎯 **Smart silence detection** - Automatically trigger transcription on pauses
- 🔒 **Privacy-focused** - All processing happens locally
- 😊 **Emotion classification** - Detects specific emotions (anger, fear, joy, sadness, etc.)
- ❓ **Question detection** - Automatic identification of questions
- 🎤 **Voice Activity Detection (VAD)** - Real-time speech vs silence detection
- 👤 **Speaker detection** - Identifies speaker changes and counts unique voices
- 📝 **Conversation logging** - Stores all conversations with metadata in SQLite database
- 🔍 **Semantic vectorization** - Vectorize conversations for advanced search and analysis
- 🧠 **AI-powered session analysis** - Generate contextual prompts using Gemma 3n

## Files

- `transcriber.py` - **Offline** real-time audio transcription with emotion classification, VAD, and logging
- `simple_voice_gemma.py` - Voice-enabled chat with Gemma AI models and conversation logging
- `gemma_runner.py` - Original text-based Gemma interface
- `conversation_logger.py` - Conversation logging and storage system
- `conversation_viewer.py` - View and analyze conversation logs
- `conversation_vectorizer.py` - Vectorize conversations for semantic search and analysis
- `session_analyzer.py` - Analyze sessions using Gemma 3n to generate contextual prompts
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

### 4. Conversation Log Management

View and analyze your conversation logs:

```bash
# List all conversation sessions
python conversation_viewer.py list

# View detailed transcript of a session
python conversation_viewer.py view --session session_20241201_143022

# Analyze emotion patterns in a session
python conversation_viewer.py analyze --session session_20241201_143022

# Export session data
python conversation_viewer.py export --session session_20241201_143022 --format json

### 5. Conversation Vectorization and Semantic Search

Vectorize your conversations for advanced search and analysis:

```bash
# Vectorize all conversation sessions
python conversation_vectorizer.py vectorize

# Vectorize a specific session
python conversation_vectorizer.py vectorize --session session_20241201_143022

# Semantic search across all conversations
python conversation_vectorizer.py search --query "how are you feeling today"

# Find utterances with specific emotions
python conversation_vectorizer.py emotion --emotion joy --top-k 10

# Analyze a specific speaker
python conversation_vectorizer.py speaker --speaker "Speaker A" --top-k 10

# View vectorization statistics
python conversation_vectorizer.py stats

### 6. Session Analysis and Contextual Prompts

Analyze conversation sessions and generate contextual prompts for future LLM interactions:

```bash
# Generate comprehensive contextual prompt
python session_analyzer.py analyze --session session_20241201_143022 --type comprehensive

# Generate emotional-focused prompt
python session_analyzer.py analyze --session session_20241201_143022 --type emotional

# Generate topic-focused prompt
python session_analyzer.py analyze --session session_20241201_143022 --type topical

# Generate speaker profile
python session_analyzer.py profile --session session_20241201_143022 --speaker "Speaker A"

# Generate conversation summary
python session_analyzer.py summary --session session_20241201_143022

# Save contextual prompt to file
python session_analyzer.py save --session session_20241201_143022 --type emotional
```

## How It Works

### Audio Transcription Flow

1. **Audio Capture**: Continuously captures audio from your microphone
2. **Silence Detection**: Monitors audio levels to detect speech pauses
3. **Buffer Management**: Maintains a rolling buffer of recent audio
4. **Transcription Trigger**: When silence is detected, processes the audio buffer
5. **Speech Recognition**: Uses Vosk (offline) to convert audio to text
6. **Voice Analysis**: VAD and speaker detection
7. **Text Analysis**: Analyzes emotions and detects questions
8. **Conversation Logging**: Stores all data in SQLite database and JSON files
9. **Vectorization**: Creates semantic embeddings for advanced search capabilities
10. **Real-time Output**: Displays transcribed text with speaker, emotion, and question indicators

### Voice Chat Flow

1. **Audio Transcription**: Captures and transcribes your speech
2. **Voice Analysis**: VAD and speaker detection
3. **Text Analysis**: Analyzes emotions and detects questions
4. **Text Processing**: Sends transcribed text to Gemma model
5. **AI Response**: Receives and displays Gemma's response
6. **Conversation Logging**: Stores all data in SQLite database and JSON files
7. **Vectorization**: Creates semantic embeddings for advanced search capabilities
8. **Continuous Loop**: Repeats for natural conversation flow

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