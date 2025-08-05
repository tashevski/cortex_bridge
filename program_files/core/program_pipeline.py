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
from ai.adaptive_system_monitor import adaptive_monitor, SystemMode
from utils.ollama_utils import ensure_ollama_running, ensure_required_models
from config.config import cfg
from .pipeline_helpers import handle_gemma_response, print_speaker_info, process_feedback, handle_special_commands

def load_vosk_model(config=None):
    """Load Vosk model using configuration"""
    if config is None:
        config = cfg.vosk_model
    
    print("Loading Vosk model...")
    
    # Get base directory for models
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(current_dir, config.models_base_dir)
    
    # Try preferred models in order
    for model_type in config.preferred_models:
        if model_type in config.available_models:
            model_info = config.available_models[model_type]
            model_path = os.path.join(models_dir, model_info["name"])
            
            if os.path.exists(model_path):
                print(f"   Using: {model_info['name']} ({model_info['accuracy']} accuracy)")
                return Model(model_path)
            else:
                print(f"   Model {model_info['name']} not found at {model_path}")
    
    # Fallback to configured fallback model
    fallback_path = os.path.join(models_dir, config.fallback_model_name)
    print(f"   Using fallback: {config.fallback_model_name}")
    return Model(fallback_path)

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
        # Set mode to processing while handling feedback
        adaptive_monitor.set_system_mode(SystemMode.PROCESSING, "Processing user feedback")
        
        feedback = process_feedback(text)
        if conversation_manager.vector_db:
            conversation_manager.vector_db.update_session_with_feedback(
                conversation_manager.session_id, feedback)
        
        conversation_manager.reset_conversation()
        conversation_manager.start_new_conversation()
        print("🎤 Back to listening mode")
        
        # Return to listening mode
        adaptive_monitor.set_system_mode(SystemMode.LISTENING, "Ready for next input")
        return
    
    if conversation_manager.in_gemma_mode:
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)
        
        if conversation_manager.should_exit_gemma_mode(text):
            print("Was that helpful?")
            conversation_manager.waiting_for_feedback = True
            # Stay in listening mode for feedback
            adaptive_monitor.set_system_mode(SystemMode.LISTENING, "Waiting for feedback")
            return
        
        # Set mode to GEMMA before processing
        adaptive_monitor.set_system_mode(SystemMode.GEMMA, "Processing LLM request")
        
        context = conversation_manager.get_conversation_context()
        handle_gemma_response(gemma_client, text, context, conversation_manager)
        
        # Return to listening after LLM response
        adaptive_monitor.set_system_mode(SystemMode.LISTENING, "LLM response complete")
        return
    
    if conversation_manager.should_enter_gemma_mode(text, emotion_text, confidence):
        print("🤖 Entering Gemma conversation mode...")
        conversation_manager.start_new_conversation()
        conversation_manager.in_gemma_mode = True
        
        if conversation_manager.is_question(text):
            conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)
        
        # Set mode to GEMMA for initial processing
        adaptive_monitor.set_system_mode(SystemMode.GEMMA, "Entering conversation mode")
        
        handle_gemma_response(gemma_client, text, "", conversation_manager)
        
        # Return to listening after initial response
        adaptive_monitor.set_system_mode(SystemMode.LISTENING, "Conversation mode active")
    else:
        print("⏭️  Not a question - staying in listening mode")
        # Save listening mode conversations too!
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features, emotion_text, confidence)
        
        # Ensure we're in listening mode
        adaptive_monitor.set_system_mode(SystemMode.LISTENING, "Processing non-question input")

def main():
    """Main speech processing pipeline"""
    print("🎤 Speech Processing Pipeline - Speak (Ctrl+C to stop)")
    print("📊 Questions will enter Gemma conversation mode")
    print("🤖 Adaptive monitoring enabled")
    
    # Initialize adaptive monitoring
    adaptive_monitor.set_system_mode(SystemMode.IDLE, "System starting up")
    adaptive_monitor.start_monitoring()
    
    if not ensure_ollama_running() or not ensure_required_models():
        print("❌ Failed to initialize Ollama. Exiting.")
        adaptive_monitor.set_system_mode(SystemMode.SHUTDOWN, "Ollama initialization failed")
        adaptive_monitor.stop_monitoring()
        return
    
    print("✅ Ollama initialization complete")
    
    model = load_vosk_model()
    emotion_classifier = EmotionClassifier() # Load emotion classifier 
    conversation_manager = ConversationManager()
    speech_processor = SpeechProcessor()  # Uses config defaults
    gemma_client = OptimizedGemmaClient()  # Uses config defaults
    
    speaker_detector = SpeakerDetector(enhanced_db=conversation_manager.vector_db)  # Uses config defaults
    
    # Start initial session for listening mode
    conversation_manager.start_new_conversation()
    
    # Set to listening mode after initialization
    adaptive_monitor.set_system_mode(SystemMode.LISTENING, "Ready for speech input")
    
    sample_rate = cfg.vosk_model.sample_rate
    rec = KaldiRecognizer(model, sample_rate)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, 
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
                # Set to processing mode during active speech processing
                if adaptive_monitor.get_system_mode() == SystemMode.LISTENING:
                    adaptive_monitor.set_system_mode(SystemMode.PROCESSING, "Processing speech input")
                speaker_detector.update_speaker_count(data, speech_processor.silence_frames)
            else:
                # Return to listening mode when no speech detected
                if adaptive_monitor.get_system_mode() == SystemMode.PROCESSING:
                    adaptive_monitor.set_system_mode(SystemMode.LISTENING, "No speech detected")
            
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
                    rec = KaldiRecognizer(model, sample_rate)
            else:
                frames_with_same_speaker = 0
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '').strip()
                if not text:
                    continue
                
                if text.lower() == "exit program":
                    print("ending program")
                    adaptive_monitor.set_system_mode(SystemMode.SHUTDOWN, "User requested exit")
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
        adaptive_monitor.set_system_mode(SystemMode.SHUTDOWN, "Keyboard interrupt")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        adaptive_monitor.stop_monitoring()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()

