# Vector Context Integration for Gemma Responses

## Overview

Minimal integration that automatically adds relevant vector database context to Gemma responses. When enabled, the system searches for similar conversations, cue cards, and adaptive prompts to enhance response quality.

## How It Works

1. **Query Processing**: When a user asks a question, the system searches the vector database for relevant content
2. **Context Retrieval**: Finds similar cue cards and adaptive prompts based on semantic similarity
3. **Response Enhancement**: Adds retrieved context to the Gemma prompt for more informed responses

## Configuration

Enable/disable in `program_files/config/config.py`:

```python
class ConversationModeConfig:
    use_vector_context = True  # Set to False to disable
```

## What Gets Retrieved

- **Cue Cards**: Question-answer pairs from medical documents
- **Adaptive Prompts**: Medical issue-specific prompts and responses
- **Conversations**: Similar past conversations (if available)

## Usage

The integration is automatic when enabled. No additional code needed - just ask questions normally and the system will include relevant vector context.

## Example

**User Query**: "How should I manage diabetes?"

**Vector Context Added**:
```json
{
  "relevant_cue_cards": [
    {"q": "How to manage diabetes?", "a": "Monitor blood sugar regularly..."}
  ],
  "relevant_prompts": [
    {"issue": "diabetes", "prompt": "briefly summarise diabetes issues..."}
  ]
}
```

**Result**: Gemma gets enhanced context for more informed responses.

## Performance

- Minimal latency impact (vector search is fast)
- Automatic fallback if no relevant content found
- Graceful error handling if database unavailable 