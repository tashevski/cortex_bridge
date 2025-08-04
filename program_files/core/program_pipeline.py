#!/usr/bin/env python3
"""Simplified Speech Processing Pipeline"""

import json
import pyaudio
import os
from typing import Optional, Dict
from vosk import Model, KaldiRecognizer
from .conversation_manager import ConversationManager
from speech.speech_processor import SpeechProcessor, SpeakerDetector
from ai.gemma_client import GemmaClient
from utils.text_utils import is_question
from utils.ollama_utils import ensure_ollama_running, ensure_required_models

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

def process_text(text: str, conversation_manager: ConversationManager, gemma_client: GemmaClient, 
                speaker_detector, audio_features: Optional[Dict] = None):
    """Process transcribed text based on conversation state"""
    
    if conversation_manager.waiting_for_feedback:
        feedback = {"helpful": "unknown"}
        text_lower = text.lower().strip()
        
        if text_lower in ['yes', 'y', 'helpful']:
            feedback["helpful"] = True
        elif text_lower in ['no', 'n', 'not helpful']:
            feedback["helpful"] = False
        elif text_lower in ['partially', 'somewhat']:
            feedback["helpful"] = "partial"
        
        if conversation_manager.vector_db:
            conversation_manager.vector_db.update_session_with_feedback(
                conversation_manager.session_id, feedback)
        
        conversation_manager.reset_conversation()
        conversation_manager.start_new_conversation()
        print("🎤 Back to listening mode")
        return
    
    if conversation_manager.in_gemma_mode:
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features)
        
        if conversation_manager.should_exit_gemma_mode(text):
            print("Was that helpful?")
            conversation_manager.waiting_for_feedback = True
            return
        
        context = conversation_manager.get_conversation_context()
        response = gemma_client.generate_response(text, context)
        if response:
            print(f"🤖 Gemma: {response}")
            conversation_manager.add_to_history(response, False, "Gemma")
        return
    
    # if conversation_manager.should_enter_gemma_mode(text):
    #     print("🤖 Entering Gemma conversation mode...")
    #     conversation_manager.start_new_conversation()
    #     conversation_manager.in_gemma_mode = True
        
    #     if is_question(text):
    #         conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features)
        
    #     response = gemma_client.generate_response(text)
    #     if response:
    #         print(f"🤖 Gemma: {response}")
    #         conversation_manager.add_to_history(response, False, "Gemma")
    else:
        print("⏭️  Not a question - staying in listening mode")
        # Save listening mode conversations too!
        conversation_manager.add_to_history(text, True, speaker_detector.current_speaker, audio_features)

def main():
    """Main speech processing pipeline"""
    print("🎤 Speech Processing Pipeline - Speak (Ctrl+C to stop)")
    print("📊 Questions will enter Gemma conversation mode")
    
    if not ensure_ollama_running() or not ensure_required_models():
        print("❌ Failed to initialize Ollama. Exiting.")
        return
    
    print("✅ Ollama initialization complete")
    
    model = load_vosk_model()
    conversation_manager = ConversationManager()
    speech_processor = SpeechProcessor()
    gemma_client = GemmaClient("gemma3n:e4b")
    
    speaker_detector = SpeakerDetector(enhanced_db=conversation_manager.vector_db)
    
    # Start initial session for listening mode
    conversation_manager.start_new_conversation()
    
    rec = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=2048)
    stream.start_stream()
    
    # Track partial results and speaker for change detection
    partial_text = ""
    current_speaker_for_text = speaker_detector.current_speaker
    frames_with_same_speaker = 0
    
    try:
        while True:
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError as e:
                if e.errno == -9981:
                    continue
                print(f"Audio error: {e}")
                break
            
            is_speech = speech_processor.process_frame(data)
            if is_speech:
                speaker_detector.update_speaker_count(data, speech_processor.silence_frames)
            
            # Get partial results to track ongoing speech
            partial_result = json.loads(rec.PartialResult())
            new_partial_text = partial_result.get('partial', '').strip()
            
            # Update partial text if there's new content
            if new_partial_text:
                partial_text = new_partial_text
            
            # Check if speaker changed after consistent detection
            if speaker_detector.current_speaker != current_speaker_for_text:
                frames_with_same_speaker += 1
                # After 30 frames (~1.8 seconds) of consistent new speaker with partial text
                if frames_with_same_speaker >= 30 and partial_text and len(partial_text) > 5:
                    # Process the text from the previous speaker
                    print(f"📝 {partial_text}")
                    speaker_info = f"👤 {current_speaker_for_text} | 🎙️ {speaker_detector.speaker_count} voice(s)"
                    known_speakers = speaker_detector.get_known_speakers()
                    if known_speakers:
                        speaker_info += f" | 📚 Known: {', '.join(known_speakers)}"
                    print(f"   {speaker_info}")
                    
                    audio_features = speaker_detector.get_current_features()
                    process_text(partial_text, conversation_manager, gemma_client, speaker_detector, audio_features)
                    
                    if audio_features:
                        speaker_detector.clear_feature_buffer()
                    
                    # Reset for new speaker
                    partial_text = ""
                    current_speaker_for_text = speaker_detector.current_speaker
                    frames_with_same_speaker = 0
                    rec = KaldiRecognizer(model, 16000)  # Reset recognizer
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
                
                if conversation_manager.in_gemma_mode:
                    print(f"💬 You: {text}")
                elif conversation_manager.waiting_for_feedback:
                    print(f"📝 Feedback: {text}")
                else:
                    print(f"📝 {text}")
                    known_speakers = speaker_detector.get_known_speakers()
                    speaker_info = f"👤 {speaker_detector.current_speaker} | 🎙️ {speaker_detector.speaker_count} voice(s)"
                    if known_speakers:
                        speaker_info += f" | 📚 Known: {', '.join(known_speakers)}"
                    print(f"   {speaker_info}")
                
                audio_features = speaker_detector.get_current_features()
                process_text(text, conversation_manager, gemma_client, speaker_detector, audio_features)
                
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

