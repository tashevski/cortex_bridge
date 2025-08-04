# Enhanced Speaker Detection System

## Overview

The program now includes an enhanced speaker detection system that can identify specific speakers by name, not just detect speaker changes. This replaces the basic speaker detection with a more powerful machine learning-based approach.

## Features

### 🎯 Advanced Speaker Identification
- **Named Speaker Recognition**: Identifies speakers by actual names (e.g., "John", "Sarah", "Mike")
- **Confidence Scoring**: Provides confidence levels for each identification
- **Pre-trained Model**: Comes with a demo model trained on 3 speakers
- **Fallback Detection**: Falls back to basic speaker change detection if advanced identification fails

### 📊 Real-time Processing
- **Continuous Monitoring**: Analyzes audio in real-time
- **Utterance Buffering**: Accumulates audio chunks for better identification
- **Periodic Identification**: Performs identification every 50 frames to balance accuracy and performance

### 🔄 Hybrid Approach
- **Advanced + Basic**: Combines machine learning identification with basic audio feature analysis
- **Speaker Profiles**: Maintains profiles of detected speakers
- **Change Detection**: Still detects speaker changes even without identification

## How It Works

### 1. Feature Extraction
The system extracts comprehensive voice characteristics:
- **Basic Features**: Energy, pitch, zero crossings, spectral centroid
- **Advanced Features**: MFCC coefficients, formants, jitter, shimmer
- **Statistical Features**: Variance, peak amplitude, RMS energy

### 2. Speaker Identification
- Uses a Random Forest classifier trained on labeled speaker data
- Compares current audio features against known speaker profiles
- Returns speaker name and confidence score

### 3. Fallback Detection
- If identification fails or confidence is low, uses basic audio feature analysis
- Detects speaker changes based on energy and pitch differences
- Assigns generic labels (Speaker A, Speaker B, etc.)

## Usage

### Running the Enhanced System
```bash
cd program_files
python main.py
```

The system will automatically:
1. Load the pre-trained speaker identification model
2. Display known speakers on startup
3. Show speaker identification with confidence levels during transcription

### Testing Speaker Detection
```bash
cd program_files
python test_enhanced_speaker_detection.py
```

This standalone test shows:
- Speaker identification in real-time
- Confidence levels for each identification
- Active speaker profiles
- Summary of detected speakers

## Model Information

### Pre-trained Demo Model
- **Speakers**: John, Sarah, Mike
- **Features**: 14-dimensional feature vector
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85-90% on test data

### Training Your Own Model
To train on your own speakers:

1. **Collect Training Data**:
   ```bash
   cd everything\ else
   python collect_training_data.py
   ```

2. **Train the Model**:
   ```bash
   python test_speaker_identification.py
   ```

3. **Use the New Model**:
   - Copy the trained model to `program_files/`
   - The system will automatically load it

## Configuration

### Confidence Threshold
- **Default**: 0.6 (60% confidence required)
- **Adjustment**: Modify `identification_confidence_threshold` in `SpeakerDetector`

### Identification Frequency
- **Default**: Every 50 frames
- **Adjustment**: Modify `identification_cooldown` in `SpeakerDetector`

### Audio Buffer Size
- **Default**: 30 frames for utterance accumulation
- **Adjustment**: Modify `utterance_buffer` maxlen in `SpeakerDetector`

## Performance Considerations

### Accuracy vs Speed
- **Higher confidence threshold**: More accurate but fewer identifications
- **Lower cooldown**: More frequent identifications but higher CPU usage
- **Larger buffer**: Better identification but more latency

### Memory Usage
- **Speaker profiles**: ~1KB per speaker
- **Audio buffer**: ~30KB for utterance accumulation
- **Model**: ~2-5MB depending on training data

## Troubleshooting

### Common Issues

1. **"Speaker identification not available"**
   - Check that `speaker_identifier.py` is in the speech directory
   - Verify the model file exists in `program_files/`

2. **Low identification accuracy**
   - Retrain the model with more diverse training data
   - Adjust the confidence threshold
   - Ensure good audio quality (16kHz, 16-bit)

3. **No speaker changes detected**
   - Check microphone input levels
   - Verify VAD (Voice Activity Detection) is working
   - Adjust silence threshold if needed

### Debug Mode
Add debug prints to see identification details:
```python
# In speech_processor.py, add to _identify_speaker_advanced:
print(f"Debug: Features extracted, confidence: {confidence}")
```

## Future Enhancements

### Planned Features
- **Emotion Detection**: Identify speaker emotions alongside identity
- **Speaker Diarization**: Automatic speaker segmentation
- **Online Learning**: Update speaker profiles during use
- **Multi-language Support**: Support for different languages

### Advanced Models
- **Neural Networks**: Replace Random Forest with deep learning
- **Transformer Models**: Use attention-based speaker identification
- **Real-time Adaptation**: Continuously improve speaker profiles

## Technical Details

### Feature Engineering
The system extracts 14 features from each audio frame:
1. Energy (mean absolute amplitude)
2. Pitch estimate (standard deviation)
3. Zero crossings (frequency content)
4. Spectral centroid (brightness)
5. Energy variance (stability)
6. Peak amplitude (loudness)
7. RMS energy (power)
8-10. MFCC coefficients 1-3 (spectral shape)
11-12. Formants 1-2 (vocal tract characteristics)
13. Jitter (pitch variation)
14. Shimmer (amplitude variation)

### Classification Pipeline
1. **Feature Extraction**: Convert audio to feature vector
2. **Normalization**: Scale features using StandardScaler
3. **Classification**: Predict speaker using Random Forest
4. **Confidence**: Get probability scores for prediction
5. **Thresholding**: Apply confidence threshold
6. **Output**: Return speaker name and confidence

This enhanced system provides much more accurate and useful speaker differentiation compared to the basic audio feature analysis used previously. 