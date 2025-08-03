#!/usr/bin/env python3
"""Program Pipeline with Conditional Gemma Integration"""

import json
import pyaudio
import os
from vosk import Model, KaldiRecognizer
from .conditional_gemma_input import ConditionalGemmaPipeline, CONDITIONS
from .conversation_manager import ConversationManager
from speech.speech_processor import SpeechProcessor, SpeakerDetector
from ai.gemma_client import GemmaClient
from utils.utils import is_question

# Load Vosk model
print("Loading Vosk model...")
# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(current_dir, "models", "vosk-model-small-en-us-0.15")
model = Model(model_path)

# Initialize components
conditional_pipeline = ConditionalGemmaPipeline(
    model="gemma3n:e2b", 
    conditions=CONDITIONS['questions_only']
)
gemma_client = GemmaClient("gemma3n:e4b")

def handle_gemma_conversation(text: str, conversation_manager: ConversationManager, speaker_detector) -> None:        

    """Handle conversation in Gemma mode"""
    conversation_manager.add_to_history(text, is_user=True, speaker_name=speaker_detector.current_speaker)
    
    # Check if we should exit Gemma mode and ask for feedback
    if conversation_manager.should_exit_gemma_mode(text):       
        print("Was that helpful?")
        conversation_manager.waiting_for_feedback = True
        return
    
    # Continue conversation with Gemma
    context = conversation_manager.get_conversation_context()
    response = gemma_client.generate_response(text, context)
    
    if response:
        print(f"🤖 Gemma: {response}")
        conversation_manager.add_to_history(response, is_user=False, speaker_name="Gemma")
    else:
        print("❌ Failed to get response from Gemma")

def handle_feedback_mode(text: str, conversation_manager: ConversationManager) -> None:
    """Handle feedback collection mode"""
    # Create feedback object
    feedback = {
        "helpful": None,
        "response": text
    }
    
    # Categorize the response
    text_lower = text.lower().strip()
    if text_lower in ['yes', 'y', 'very helpful', 'helpful', 'it was helpful']:
        feedback["helpful"] = True
    elif text_lower in ['no', 'n', 'not helpful', 'unhelpful', 'it was not helpful']:
        feedback["helpful"] = False
    elif text_lower in ['partially', 'somewhat', 'kinda', 'sort of', 'a little']:
        feedback["helpful"] = "partial"
    else:
        feedback["helpful"] = "unknown"
    
    # Update all messages in this session with feedback
    if conversation_manager.vector_db:
        conversation_manager.vector_db.update_session_with_feedback(
            conversation_manager.session_id, feedback
        )
    
    # Store feedback for potential future use
    conversation_manager.last_feedback = feedback
    
    # Reset and return to listening mode
    conversation_manager.reset_conversation()
    print("🎤 Back to listening mode")
    
    # Start new conversation for next interaction
    conversation_manager.start_new_conversation()
    
    return feedback

def handle_listening_mode(text: str, conversation_manager: ConversationManager, speaker_detector) -> None:
    """Handle conversation in listening mode"""
    if conversation_manager.should_enter_gemma_mode(text):
        print("🤖 Entering Gemma conversation mode...")
        
        # Start new conversation session
        conversation_manager.start_new_conversation()
        
        conversation_manager.in_gemma_mode = True
        
        # Only save the message if it's a meaningful question, not just a trigger
        if is_question(text):  # Only save actual questions
            conversation_manager.add_to_history(text, is_user=True, speaker_name=speaker_detector.current_speaker)
        
        # Initial response from Gemma
        response = gemma_client.generate_response(text)
        if response:
            print(f"🤖 Gemma: {response}")
            conversation_manager.add_to_history(response, is_user=False, speaker_name="Gemma")
    else:
        print("⏭️  Not a question - staying in listening mode")

def main():
    """Main transcription and conditional Gemma pipeline"""
    print("🎤 Conditional Gemma Pipeline - Speak (Ctrl+C to stop)")
    print("📊 Questions will enter Gemma conversation mode")
    print("💬 Say 'exit' to leave Gemma conversation mode")
    
    # Initialize components
    conversation_manager = ConversationManager()
    speech_processor = SpeechProcessor()
    speaker_detector = SpeakerDetector()
    
    # Initialize speech recognizer
    rec = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()
    
    # Open audio stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=2048)
    stream.start_stream()
    
    try:
        while True:
            # Read audio data
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError as e:
                if e.errno == -9981:
                    continue
                else:
                    print(f"Audio error: {e}")
                    break
            
            # Process VAD and speaker detection
            is_speech = speech_processor.process_frame(data)
            if is_speech:
                speaker_detector.update_speaker_count(data, speech_processor.silence_frames)
            
            # Process audio through Vosk recognizer
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    text = result['text']
                    
                    # Check for program exit
                    if text.lower() == "exit program":
                        print("ending program")
                        break
                    
                    # Display transcription
                    if conversation_manager.in_gemma_mode:
                        print(f"💬 You: {text}")
                    elif conversation_manager.waiting_for_feedback:
                        print(f"📝 Feedback: {text}")
                    else:
                        print(f"📝 {text}")
                        print(f"   👤 {speaker_detector.current_speaker} | 🎙️ {speaker_detector.speaker_count} voice(s)")
                    
                    # Handle conversation based on mode
                    if conversation_manager.waiting_for_feedback:
                        feedback = handle_feedback_mode(text, conversation_manager)
                    elif conversation_manager.in_gemma_mode:
                        handle_gemma_conversation(text, conversation_manager, speaker_detector)
                    else:
                        handle_listening_mode(text, conversation_manager, speaker_detector)
                        
    except KeyboardInterrupt:
        print("\n🛑 Stopping pipeline...")
        
        # Clean shutdown
        stream.stop_stream()
        stream.close()
        audio.terminate()
        conditional_pipeline.cleanup()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()

