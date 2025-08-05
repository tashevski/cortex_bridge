# RAG Functions Reorganization

## Overview
The rag_functions directory has been reorganized into logical subfolders for better separation of concerns and cleaner code architecture.

## New Subfolder Structure

```
rag_functions/
├── core/                    # Core processing functionality
│   ├── __init__.py
│   ├── main.py             # Main document processing pipeline
│   ├── llm_analysis.py     # LLM analysis functions
│   ├── config.py           # Configuration management
│   └── medical_processing.py # Medical workflow
├── templates/              # Prompt templates
│   ├── __init__.py
│   └── prompt_templates.py # Medical documentation templates
├── ml/                     # Machine learning & AI operations
│   ├── __init__.py
│   ├── vector_operations.py # Vector similarity & document analysis
│   └── cue_card_extraction.py # Medical cue card generation
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── retrieval.py        # Reference document retrieval
│   ├── semantic_parser.py  # Document parsing
│   ├── ocr_layout_copy.py  # OCR functionality
│   └── ocr_layout.py       # Legacy OCR
├── examples/               # Example usage and tests
│   ├── __init__.py
│   ├── example_prompt_usage.py
│   ├── qa_example.py
│   └── test_gemma_integration.py
├── models/                 # Model files
└── __init__.py            # Main package interface
```

## Module Descriptions

### Core (`rag_functions.core`)
- **`main.py`**: Primary document processing pipeline
- **`llm_analysis.py`**: LLM analysis and generation functions
- **`config.py`**: RAG configuration management
- **`medical_processing.py`**: Complete medical document workflow

### Templates (`rag_functions.templates`)
- **`prompt_templates.py`**: Medical documentation templates
  - SOAP notes, intake forms, diagnostic reports, treatment plans
  - Template selection utilities

### ML (`rag_functions.ml`)
- **`vector_operations.py`**: Vector similarity operations
  - Template selection via vector similarity
  - Document type analysis
  - Sentence vectorization
- **`cue_card_extraction.py`**: Medical cue card generation
  - Clustering-based extraction with Gemma
  - CueCard dataclass

### Utils (`rag_functions.utils`)
- **`retrieval.py`**: Reference document retrieval
- **`semantic_parser.py`**: Document parsing and entity extraction
- **`ocr_layout_copy.py`**: OCR text and layout extraction

### Examples (`rag_functions.examples`)
- **`example_prompt_usage.py`**: Template usage demonstrations
- **`qa_example.py`**: Q&A examples
- **`test_gemma_integration.py`**: Integration tests

## Migration Guide

### Old Imports (Flat Structure)
```python
from rag_functions.prompt_templates import (
    select_optimal_templates, extract_cue_cards,
    get_template, analyze_document_type
)
```

### New Imports (Subfolder Structure)
The main package interface remains the same for convenience:
```python
# Simple imports (recommended)
from rag_functions import (
    get_template, select_optimal_templates, extract_cue_cards,
    analyze_document_type, process_medical_document, CueCard
)

# Or import from specific modules
from rag_functions.templates import get_template
from rag_functions.ml import select_optimal_templates, extract_cue_cards
from rag_functions.core import process_medical_document
```

## Benefits

1. **Logical Organization**: Files grouped by functionality in subfolders
2. **Clear Separation**: Core, ML, templates, and utilities are distinct
3. **Better Maintainability**: Easy to locate and modify specific functionality
4. **Scalability**: Easy to add new modules within appropriate subfolders
5. **Clean Imports**: Main package interface unchanged for backward compatibility
6. **Modular Testing**: Each subfolder can be tested independently

## Functionality Preserved

All original functionality is preserved:
- ✅ Vector-based template selection with similarity thresholds
- ✅ Sentence-based clustering for cue card extraction
- ✅ Gemma integration for medical cue card generation  
- ✅ Format: "{context}: {medical/care advice}"
- ✅ ~80 lines vs 500+ originally (minimal, clean code)
- ✅ Backward compatible imports through main `__init__.py`

## Usage Examples

```python
# Same as before - no changes needed!
from rag_functions import (
    get_template, select_optimal_templates, extract_cue_cards,
    process_medical_document
)

# Template selection
templates = select_optimal_templates(content, task_description)
template = get_template(templates[0][0])

# Cue card extraction
cue_cards = extract_cue_cards(llm_output, context_type="medical")

# Complete workflow
result = process_medical_document(content, task, gemma_client)
```

## Directory Structure Benefits

- **`core/`**: All main processing logic in one place
- **`templates/`**: All prompt templates and template utilities
- **`ml/`**: All AI/ML operations (vectors, clustering, cue cards)
- **`utils/`**: All utility functions (parsing, OCR, retrieval)
- **`examples/`**: All example code and tests separate from core logic

## Migration Status
✅ **COMPLETE** - All files moved, imports updated, functionality preserved