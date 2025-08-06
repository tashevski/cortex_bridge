# Image Capability Guide

## Overview
The system now supports multimodal input, allowing you to analyze images alongside speech input using the Gemma AI model.

## How It Works

### 1. **Base Infrastructure**
- `GemmaClient.generate_response()` accepts an `image_path` parameter
- Images are automatically encoded to base64 and sent to the Ollama API
- The system supports common image formats (JPEG, PNG, etc.)

### 2. **Optimized Pipeline**
- `OptimizedGemmaClient.generate_response_optimized()` detects image presence
- Model selection automatically adjusts for multimodal input
- Latency monitoring tracks image processing performance

### 3. **Template Integration**
- Prompt templates work seamlessly with images
- The SOAP note template can analyze medical images
- Context and image are combined in the final prompt

## Usage Examples

### Basic Image Analysis
```python
# In your code, you can now pass an image path:
process_text(
    "What do you see in this X-ray?", 
    conversation_manager, 
    gemma_client, 
    speaker_detector, 
    tts_file, 
    image_path="/path/to/xray.jpg"
)
```

### Medical Image Analysis with SOAP Template
```python
# The system automatically uses the SOAP template when in Gemma mode:
# User: "Analyze this chest X-ray for pneumonia"
# System: Uses SOAP template + image for structured medical analysis
```

### Programmatic Integration
```python
# Direct API call with image:
response = gemma_client.generate_response_optimized(
    prompt="Describe this image",
    context="Medical imaging analysis",
    prompt_template=soap_template,
    image_path="/path/to/image.jpg"
)
```

## Supported Image Types
- **Medical Images**: X-rays, CT scans, MRIs, ultrasounds
- **General Images**: Photos, diagrams, charts
- **Formats**: JPEG, PNG, BMP, TIFF
- **Size**: Limited by Ollama API (typically < 10MB)

## Template Examples

### SOAP Note with Image
```
<start_of_turn>system
You are a medical documentation specialist.
<end_of_turn>

<start_of_turn>user
Patient Data: {context}
Task: {prompt}
[Image attached for analysis]

Create a SOAP note with:
SUBJECTIVE: Chief complaint, history
OBJECTIVE: Vital signs, examination, image findings
ASSESSMENT: Clinical impression based on image
PLAN: Treatment recommendations
<end_of_turn>

<start_of_turn>model
```

## Future Enhancements
- **Image Upload Interface**: Web UI for drag-and-drop image uploads
- **Batch Processing**: Analyze multiple images in sequence
- **Image Preprocessing**: Automatic resizing and format conversion
- **Specialized Models**: Model selection based on image type (medical vs. general)

## Technical Notes
- Images are encoded to base64 for API transmission
- The system automatically detects image presence and adjusts model selection
- Latency monitoring includes image processing time
- Error handling for invalid image files or encoding issues 