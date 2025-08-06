# RAG Vector Database Integration

## Overview

The RAG (Retrieval-Augmented Generation) functions now integrate with the existing ChromaDB vector database to store and retrieve cue cards and adaptive prompts. This enables persistent storage and semantic search capabilities for all generated medical advice and prompts.

## Features

### 🔄 Automatic Storage
- **Cue Cards**: Each generated cue card (question-answer pair) is automatically stored in the vector database
- **Adaptive Prompts**: Medical issue-specific prompts are stored with their corresponding medical issues
- **Rich Metadata**: Each item includes document path, timestamp, content type, and relevant metadata

### 🔍 Semantic Search
- **Query-based search**: Find relevant cue cards and prompts using natural language queries
- **Filtering**: Filter by prompt type, medical issue, or document path
- **Statistics**: Get comprehensive statistics about stored content

## Usage

### Processing Documents

When you process a document using `process_document()`, cue cards and adaptive prompts are automatically stored:

```python
from rag_functions.core.main import process_document

# Process a document - cue cards and adaptive prompts are stored automatically
result = process_document("/path/to/medical/document.pdf")

print(f"Stored {len(result['adaptive_prompts'])} adaptive prompts")
print(f"Stored cue cards for {len(result['contextual_responses'])} prompt types")
```

### Searching Cue Cards

```python
from rag_functions.utils.retrieval import search_cue_cards, get_all_cue_cards

# Search for cue cards by query
cue_cards = search_cue_cards(
    query="diabetes management", 
    top_k=10
)

# Get all cue cards for a specific prompt type
family_cards = get_all_cue_cards(
    prompt_type="medical and care advice for family"
)

# Get all cue cards from a specific document
doc_cards = get_all_cue_cards(
    document_path="/path/to/document.pdf"
)
```

### Searching Adaptive Prompts

```python
from rag_functions.utils.retrieval import search_adaptive_prompts, get_all_adaptive_prompts

# Search for adaptive prompts by query
prompts = search_adaptive_prompts(
    query="medication", 
    top_k=10
)

# Get all prompts for a specific medical issue
diabetes_prompts = get_all_adaptive_prompts(
    medical_issue="diabetes"
)

# Get all adaptive prompts from a specific document
doc_prompts = get_all_adaptive_prompts(
    document_path="/path/to/document.pdf"
)
```

### Getting Statistics

```python
from rag_functions.utils.retrieval import get_rag_stats

# Get comprehensive statistics
stats = get_rag_stats()

print(f"Total cue cards: {stats['total_cue_cards']}")
print(f"Total adaptive prompts: {stats['total_adaptive_prompts']}")
print(f"Prompt types: {stats['prompt_types']}")
print(f"Medical issues: {stats['medical_issues']}")
```

## Data Structure

### Cue Card Storage

Each cue card is stored with the following structure:

```json
{
  "content": "Question: How to manage diabetes?\nAnswer: Monitor blood sugar regularly...",
  "metadata": {
    "document_path": "/path/to/document.pdf",
    "cue_card_id": "uuid_123",
    "prompt_type": "medical and care advice for family",
    "question": "How to manage diabetes?",
    "answer": "Monitor blood sugar regularly...",
    "timestamp": "2024-01-15T10:30:00",
    "content_type": "cue_card",
    "session_id": "rag_session_2024-01-15T10-30-00"
  }
}
```

### Adaptive Prompt Storage

Each adaptive prompt is stored with the following structure:

```json
{
  "content": "briefly summarise and identify any issues relating to diabetes...",
  "metadata": {
    "document_path": "/path/to/document.pdf",
    "prompt_id": "adaptive_uuid_456",
    "medical_issue": "diabetes",
    "prompt_text": "briefly summarise and identify any issues relating to diabetes...",
    "timestamp": "2024-01-15T10:30:00",
    "content_type": "adaptive_prompt",
    "session_id": "rag_session_2024-01-15T10-30-00"
  }
}
```

## Database Location

The vector database is stored in the same location as the main program's conversation database:

```
program_files/
  └── data/
      └── vector_db/
          ├── chroma.sqlite3
          └── [collection folders]
```

## Testing

Run the test script to verify functionality:

```bash
python test_rag_vector_db.py
```

This will:
- Display statistics about stored content
- Search for cue cards and adaptive prompts
- Show examples of retrieved data

## Integration with Main Program

The RAG vector database uses the same ChromaDB instance as the main conversation system, ensuring:

- **Unified storage**: All vector data in one location
- **Consistent access**: Same database client and configuration
- **Shared persistence**: Data survives program restarts
- **Compatible metadata**: Consistent structure across all stored items

## Error Handling

The system gracefully handles:
- **Missing dependencies**: Falls back gracefully if ChromaDB is not available
- **Database errors**: Provides informative error messages
- **Empty results**: Returns empty lists instead of failing
- **Import issues**: Graceful degradation when vector database is not accessible

## Performance Considerations

- **Semantic search**: Uses ChromaDB's built-in embedding and similarity search
- **Efficient filtering**: Metadata-based filtering for fast queries
- **Batch operations**: Multiple items stored efficiently
- **Persistent storage**: No need to reprocess documents

## Future Enhancements

Potential improvements:
- **Advanced filtering**: Date ranges, confidence scores
- **Bulk operations**: Import/export functionality
- **Analytics**: Usage patterns and effectiveness metrics
- **Integration**: Direct integration with conversation system for context-aware responses 