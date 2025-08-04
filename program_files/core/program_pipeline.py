#!/usr/bin/env python3
"""Simplified Speech Processing Pipeline"""

import json
import pyaudio
import os
from typing import Optional, Dict
from vosk import Model, KaldiRecognizer
from .conversation_manager import ConversationManager
from speech.speech_processor import SpeechProcessor, SpeakerDetector
from ai.optimized_gemma_client import OptimizedGemmaClient
from utils.ollama_utils import ensure_ollama_running, ensure_required_models
from .pipeline_helpers import handle_gemma_response, print_speaker_info, process_feedback, handle_special_commands

def load_vosk_model():
    print("Loading Vosk model...")
    try:
        from config.vosk_config import get_vosk_model_path, get_vosk_model_info
        model_path = get_vosk_model_path()
        model_info = get_vosk_model_info()
        print(f"   Using: {model_info['name']} ({model_info['accuracy']} accuracy)")
    except ImportError:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, "models", "vosk-model-en-us-0.22")
        print("   Using: vosk-model-en-us-0.22 (fallback)")
    return Model(model_path)

class EmotionClassifier:
    """Emotion classification using a HuggingFace pipeline."""

    def __init__(self):
        print("Loading emotion classification model...")
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )

    def process(self, text: str):
        """Return the top emotion label and confidence for the given text."""
        if self.classifier is None:
            return "neutral", 0.0
        
        try:
            emotions = self.classifier(text)[0]
            top_emotion = max(emotions, key=lambda x: x['score'])
            emotion_text = f"{top_emotion['label'].title()}"
            confidence = top_emotion['score']
            return emotion_text, confidence
        except Exception as e:
            print(f"⚠️  Emotion processing failed: {e}")
            return "neutral", 0.0

def process_text(text: str, conversation_manager: ConversationManager, gemma_client: OptimizedGemmaClient, 
                speaker_detector, audio_features: Optional[Dict] = None, emotion_text: str = None, confidence: float = None):
    """Process transcribed text based on conversation state"""
    
    if conversation_manager.waiting_for_feedback:
        feedback = process_feedback(text)
        if conversation_manager.vector_db:
            conversation_manager.vector_db.update_session_with_feedback(
                conversation_manager.session_id, feedback)
        
        conversation_manager.reset_conversation()
        conversation_manager.start_new_conversation()
        print("🎤 Back to listening mode")
        return
    
    if conversation_manager.in_gemma_mode:
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)
        
        if conversation_manager.should_exit_gemma_mode(text):
            print("Was that helpful?")
            conversation_manager.waiting_for_feedback = True
            return
        
        context = conversation_manager.get_conversation_context()
        handle_gemma_response(gemma_client, text, context, conversation_manager)
        return
    
    if conversation_manager.should_enter_gemma_mode(text):
        print("🤖 Entering Gemma conversation mode...")
        conversation_manager.start_new_conversation()
        conversation_manager.in_gemma_mode = True
        
        if conversation_manager.is_question(text):
            conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)
        
        handle_gemma_response(gemma_client, text, "", conversation_manager)
    else:
        print("⏭️  Not a question - staying in listening mode")
        # Save listening mode conversations too!
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)

def main():
    """Main speech processing pipeline"""
    print("🎤 Speech Processing Pipeline - Speak (Ctrl+C to stop)")
    print("📊 Questions will enter Gemma conversation mode")
    
    if not ensure_ollama_running() or not ensure_required_models():
        print("❌ Failed to initialize Ollama. Exiting.")
        return
    
    print("✅ Ollama initialization complete")
    
    model = load_vosk_model()
    emotion_classifier = EmotionClassifier() # Load emotion classifier 
    conversation_manager = ConversationManager()
    speech_processor = SpeechProcessor()
    gemma_client = OptimizedGemmaClient()  # Uses config defaults
    
    speaker_detector = SpeakerDetector(enhanced_db=conversation_manager.vector_db)
    
    # Start initial session for listening mode
    conversation_manager.start_new_conversation()
    
    rec = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=2048)
    stream.start_stream()
    
    # Track speaker changes for message segmentation
    partial_text = ""
    current_speaker_for_text = speaker_detector.current_speaker
    frames_with_same_speaker = 0
    
    def process_message(text, speaker):
        """Process a complete message"""
        print(f"📝 {text}")
        known_speakers = speaker_detector.get_known_speakers()
        print_speaker_info(speaker, speaker_detector.speaker_count, known_speakers)
        
        audio_features = speaker_detector.get_current_features()
        # We skip emotion classification here to avoid duplicate costly inference.
        process_text(text, conversation_manager, gemma_client, speaker_detector, audio_features)
        if audio_features:
            speaker_detector.clear_feature_buffer()
    
    try:
        while True:
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError as e:
                if e.errno == -9981:
                    continue
                print(f"Audio error: {e}")
                break
            
            # Process speech and speaker detection
            is_speech = speech_processor.process_frame(data)
            
            # Record speech activity for latency monitoring
            gemma_client.record_speech_activity(is_speech)
            
            if is_speech:
                speaker_detector.update_speaker_count(data, speech_processor.silence_frames)
            
            # Track partial results for speaker change detection
            partial_result = json.loads(rec.PartialResult())
            new_partial_text = partial_result.get('partial', '').strip()
            if new_partial_text:
                partial_text = new_partial_text
            
            # Check for speaker change based message segmentation
            if speaker_detector.current_speaker != current_speaker_for_text:
                frames_with_same_speaker += 1
                if frames_with_same_speaker >= 30 and partial_text and len(partial_text) > 5:
                    process_message(partial_text, current_speaker_for_text)
                    partial_text = ""
                    current_speaker_for_text = speaker_detector.current_speaker
                    frames_with_same_speaker = 0
                    rec = KaldiRecognizer(model, 16000)
            else:
                frames_with_same_speaker = 0
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '').strip()
                if not text:
                    continue
                
                if text.lower() == "exit program":
                    print("ending program")
                    break
                
                if handle_special_commands(text, gemma_client, conversation_manager):
                    continue
                
                if conversation_manager.in_gemma_mode:
                    print(f"💬 You: {text}")
                elif conversation_manager.waiting_for_feedback:
                    print(f"📝 Feedback: {text}")
                else:
                    print(f"📝 {text}")
                    known_speakers = speaker_detector.get_known_speakers()
                    print_speaker_info(speaker_detector.current_speaker, speaker_detector.speaker_count, known_speakers)
                
                audio_features = speaker_detector.get_current_features()
                # Determine emotion for full recognized text
                emotion_text, confidence = emotion_classifier.process(text)
                print(f"🎭 Emotion: {emotion_text} (Confidence: {confidence:.2f})")
                process_text(text, conversation_manager, gemma_client, speaker_detector, audio_features, emotion_text, confidence)
                
                if audio_features:
                    speaker_detector.clear_feature_buffer()
                
                # Reset tracking after normal message completion
                partial_text = ""
                current_speaker_for_text = speaker_detector.current_speaker
                frames_with_same_speaker = 0
                        
    except KeyboardInterrupt:
        print("\n🛑 Stopping pipeline...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()

