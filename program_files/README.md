# Program Files - Organized Structure

This directory contains the refactored and organized speech processing pipeline with AI integration.

## Folder Structure

```
program_files/
├── main.py                    # Main entry point
├── core/                      # Core pipeline components
│   ├── __init__.py
│   ├── program_pipeline.py    # Main orchestration
│   ├── conversation_manager.py # Conversation state management
│   └── conditional_gemma_input.py # Conditional routing logic
├── speech/                    # Speech processing components
│   ├── __init__.py
│   └── speech_processor.py    # VAD and speaker detection
├── ai/                        # AI integration components
│   ├── __init__.py
│   ├── gemma_client.py        # Gemma API client
│   └── gemma_runner.py        # Standalone Gemma runner
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── utils.py               # Shared helper functions
├── models/                    # Model files
│   └── vosk-model-small-en-us-0.15/ # Vosk speech recognition model
└── docs/                      # Documentation
    ├── PROGRAM_WORKFLOW_GUIDE.md
    ├── REFACTORING_SUMMARY.md
    └── summary.pdf
```

## Quick Start

### Run the main pipeline:
```bash
python main.py
```

### Run standalone Gemma chat:
```bash
python -m ai.gemma_runner --model gemma3n:e4b
```

## Component Overview

### Core Components (`core/`)
- **program_pipeline.py**: Main orchestration and audio processing loop
- **conversation_manager.py**: Manages conversation states and history
- **conditional_gemma_input.py**: Determines when to route text to AI

### Speech Processing (`speech/`)
- **speech_processor.py**: Voice Activity Detection (VAD) and speaker identification

### AI Integration (`ai/`)
- **gemma_client.py**: Simplified Gemma API interactions
- **gemma_runner.py**: Standalone Gemma chat interface

### Utilities (`utils/`)
- **utils.py**: Shared helper functions (question detection, keyword checking, etc.)

### Models (`models/`)
- **vosk-model-small-en-us-0.15/**: Vosk speech recognition model

### Documentation (`docs/`)
- **PROGRAM_WORKFLOW_GUIDE.md**: Complete workflow explanation
- **REFACTORING_SUMMARY.md**: Refactoring improvements summary
- **summary.pdf**: Additional documentation

## Import Examples

```python
# Import core components
from core import ConversationManager, ConditionalGemmaPipeline

# Import speech processing
from speech import SpeechProcessor, SpeakerDetector

# Import AI components
from ai import GemmaClient, GemmaRunner

# Import utilities
from utils import is_question, contains_keywords
```

## Benefits of This Organization

1. **Clear Separation of Concerns**: Each folder has a specific purpose
2. **Easy Navigation**: Related files are grouped together
3. **Modular Imports**: Clean import statements with package structure
4. **Scalability**: Easy to add new components to appropriate folders
5. **Maintainability**: Clear structure makes code easier to maintain
6. **Documentation**: All docs are organized in one place

## Migration Notes

- All import statements have been updated to reflect the new structure
- The main entry point is now `main.py` in the root directory
- Each folder is a proper Python package with `__init__.py` files
- Backward compatibility is maintained through the main entry point 