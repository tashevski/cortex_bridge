# Vosk Model Upgrade Guide

## Overview

This guide explains how to upgrade from the small Vosk model to larger, more accurate models for better speech-to-text recognition.

## Current Model

Your system currently uses **vosk-model-small-en-us-0.15** (45MB), which provides basic speech recognition but may struggle with:
- Complex words or phrases
- Accented speech
- Background noise
- Technical terminology
- Fast speech

## Available Larger Models

### 1. Medium Model (Recommended)
- **Name**: vosk-model-en-us-0.22
- **Size**: 1.5GB
- **Accuracy**: Good
- **Speed**: Medium
- **Best for**: Most users - good balance of accuracy and performance

### 2. Large Model
- **Name**: vosk-model-en-us-0.42
- **Size**: 3GB
- **Accuracy**: Excellent
- **Speed**: Slower
- **Best for**: Maximum accuracy when performance isn't critical

### 3. Large Model with Language Model
- **Name**: vosk-model-en-us-0.42-lgraph
- **Size**: 3GB
- **Accuracy**: Excellent
- **Speed**: Slower
- **Best for**: Better context understanding and word prediction

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Run the setup script**:
   ```bash
   cd program_files
   python setup_larger_vosk_model.py
   ```

2. **Choose your model**:
   - Enter `medium` for good balance (recommended)
   - Enter `large` for maximum accuracy
   - Enter `large_lgraph` for best context understanding

3. **Wait for download**:
   - Medium model: ~1.5GB download
   - Large models: ~3GB download
   - Extraction will happen automatically

### Option 2: Manual Management

1. **List available models**:
   ```bash
   python manage_vosk_models.py list
   ```

2. **Check current model**:
   ```bash
   python manage_vosk_models.py current
   ```

3. **Switch to a different model**:
   ```bash
   python manage_vosk_models.py switch medium
   ```

## Model Comparison

| Model | Size | Accuracy | Speed | Memory | Use Case |
|-------|------|----------|-------|--------|----------|
| Small | 45MB | Basic | Fast | Low | Development/testing |
| Medium | 1.5GB | Good | Medium | Medium | **General use** |
| Large | 3GB | Excellent | Slower | High | High accuracy needed |
| Large+LGraph | 3GB | Excellent | Slower | High | Best context understanding |

## Performance Impact

### Memory Usage
- **Small**: ~50MB RAM
- **Medium**: ~200MB RAM
- **Large**: ~500MB RAM

### Processing Speed
- **Small**: Fastest transcription
- **Medium**: Slightly slower, but still real-time
- **Large**: May have slight delays on slower systems

### Accuracy Improvements
- **Small → Medium**: 15-25% improvement
- **Medium → Large**: 10-15% additional improvement
- **Large → Large+LGraph**: Better context, fewer transcription errors

## Troubleshooting

### Model Not Found
If you get "Model not found" errors:

1. **Check if model is downloaded**:
   ```bash
   python manage_vosk_models.py list
   ```

2. **Download missing model**:
   ```bash
   python setup_larger_vosk_model.py
   ```

### Performance Issues
If the larger model is too slow:

1. **Switch back to smaller model**:
   ```bash
   python manage_vosk_models.py switch small
   ```

2. **Try medium model instead**:
   ```bash
   python manage_vosk_models.py switch medium
   ```

### Memory Issues
If you experience memory problems:

1. **Close other applications** to free up RAM
2. **Use medium model** instead of large
3. **Restart the application** after switching models

## Configuration Files

The system uses these configuration files:
- `config/vosk_model_config.json` - Current model selection
- `models/` - Downloaded model files

## Manual Model Download

If the automated setup doesn't work, you can manually download models:

1. **Visit**: https://alphacephei.com/vosk/models
2. **Download** your chosen model (ZIP file)
3. **Extract** to `program_files/models/`
4. **Update configuration** using the management script

## Recommendations

### For Most Users
- Start with the **medium model** (vosk-model-en-us-0.22)
- Provides significant accuracy improvement over small
- Reasonable memory and performance impact

### For High Accuracy Needs
- Use **large model** (vosk-model-en-us-0.42)
- Best for professional transcription
- Requires more system resources

### For Context-Sensitive Applications
- Use **large with language model** (vosk-model-en-us-0.42-lgraph)
- Better understanding of context and word relationships
- Best for conversational AI applications

## Testing Your Upgrade

After upgrading, test the improved accuracy:

1. **Run your speech recognition system**
2. **Try difficult words** that the small model struggled with
3. **Test with background noise**
4. **Try technical terminology**
5. **Monitor performance** to ensure it meets your needs

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your system has enough disk space and RAM
3. Ensure you have a stable internet connection for downloads
4. Check the logs for specific error messages 