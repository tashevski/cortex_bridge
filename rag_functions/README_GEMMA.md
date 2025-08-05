# RAG Functions with Gemma

## 🚀 Quick Start

### Prerequisites
1. Ensure Ollama is running: `ollama serve`
2. Pull the model: `ollama pull gemma3n:e4b`

### Basic Usage
```bash
# Using default settings
python main.py

# Skip reference documents for faster processing
python main.py --no-references

# Use specific prompt template
python main.py --template executive_summary

# Disable prompt templates (use basic prompts)
python main.py --no-template

# Enable verbose output
python main.py --verbose
```

## 🔧 Configuration

### Custom Configuration

```python
from rag_functions.config import RAGConfig

custom_config = RAGConfig(
    max_reference_chunks=10,
    request_timeout=90,
    default_template="structured_analysis",
    verbose=True
)

output = process_document(file_path, references, meta, custom_config)
```

### Prompt Templates

```python
from rag_functions.prompt_templates import get_template, create_custom_template

# Use predefined template
template = get_template("executive_summary")

# Create custom template
custom_template = create_custom_template(
    context_prefix="Research Data",
    instructions="You are a research analyst",
    output_format="Use numbered sections"
)

# Use with config
config.custom_template = custom_template
```

## 🤖 Gemma Client Features

### Basic Client
- Direct model specification
- Simple request/response
- Configurable timeout
- Image support for multimodal analysis

### Optimized Client
- Automatic model selection based on prompt
- Latency monitoring and adjustment
- Smart model switching
- Context-aware optimization

## 🔍 Advanced Features

### Vector Context
The system passes metadata to help the optimized client choose appropriate models:
- `precedent_count`: Number of precedent chunks
- `has_calculations`: Whether calculations are included
- `prefer_fast`: Speed preference flag

### Error Handling
- Automatic fallback if Ollama server is not available
- Graceful degradation with informative messages
- Timeout protection for long-running requests

### Intermediate Results
When `save_intermediate_results=True`:
- `document.txt`: Raw extracted text
- `parsed.txt`: Parsed entities
- `report_[preset].txt`: Final analysis report

## 🎯 Model Selection Guide

- **gemma:2b**: Fast, lightweight analysis
- **gemma3n:e4b**: Balanced performance and quality
- **gemma:7b**: High-quality, detailed analysis
- **Custom fine-tuned**: Use your own fine-tuned Gemma models

## 🐛 Troubleshooting

1. **"Failed to generate response from Gemma"**
   - Ensure Ollama is running: `ollama serve`
   - Check model is installed: `ollama list`

2. **Slow response times**
   - Use `--config fast` preset
   - Reduce `max_precedent_chunks` in config
   - Use smaller model like `gemma:2b`

3. **Out of memory**
   - Close other applications
   - Use smaller model
   - Reduce chunk size in config

## 📚 Example Integration

```python
from rag_functions.main import process_legal_document
from rag_functions.config import get_config

# Use quality preset for important documents
config = get_config("quality")
report = process_legal_document(
    "important_document.pdf",
    precedent_texts=["precedent1...", "precedent2..."],
    rag_config=config
)
```

## 🔄 Migration from OpenAI

The system maintains the same interface, so existing code only needs minimal changes:

```python
# Old (OpenAI)
report = analyze_with_llm(parsed, calc, precedents)

# New (Gemma)
report = analyze_with_llm(parsed, calc, precedents, config)
# or just use defaults:
report = analyze_with_llm(parsed, calc, precedents)
```