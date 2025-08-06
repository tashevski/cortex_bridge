#!/usr/bin/env python3
"""Helper functions for program_pipeline.py to reduce complexity"""

from typing import Dict, Optional
import json

def get_vector_context(query: str, conversation_context: str = "", top_k: int = 3) -> Optional[Dict]:
    """Get relevant vector context from database"""
    try:
        from rag_functions.utils.retrieval import search_cue_cards, search_adaptive_prompts
        
        # Search for relevant content
        cue_cards = search_cue_cards(query, top_k=top_k)
        adaptive_prompts = search_adaptive_prompts(query, top_k=top_k)
        
        if not cue_cards and not adaptive_prompts:
            return None
        
        return {
            "relevant_cue_cards": [{"q": c["metadata"].get("question", ""), "a": c["metadata"].get("answer", "")} for c in cue_cards],
            "relevant_prompts": [{"issue": p["metadata"].get("medical_issue", ""), "prompt": p["content"]} for p in adaptive_prompts]
        }
    except:
        return None

def handle_gemma_response(gemma_client, text: str, context: str, conversation_manager, tts_file=None, prompt_template=None, image_path=None, use_vector_context=True):
    """Generate and handle Gemma response with latency tracking and TTS"""
    
    # Get vector context if enabled
    vector_context = None
    if use_vector_context:
        vector_context = get_vector_context(text, context)
    
    response = gemma_client.generate_response_optimized(text, context, prompt_template=prompt_template, image_path=image_path, vector_context=vector_context)
    if response:
        print(f"🤖 Gemma: {response}")
        latency_metrics = gemma_client.get_last_latency_metrics()
        conversation_manager.add_to_history(response, False, "Gemma", latency_metrics=latency_metrics)
        
        # Convert response to speech using streaming TTS
        if tts_file:
            try:
                # Clean the response text before TTS processing
                cleaned_response = response.strip()
                
                # Use streaming TTS for better responsiveness
                print("🔊 Streaming response to speech...")
                if not tts_file.stream_text_to_speech(cleaned_response, chunk_length=80):
                    print("❌ Failed to stream speech for response")
            except Exception as e:
                print(f"❌ TTS error: {e}")
    
    return response

def print_speaker_info(speaker: str, speaker_count: int, known_speakers: list):
    """Print formatted speaker information"""
    info = f"👤 {speaker} | 🎙️ {speaker_count} voice(s)"
    if known_speakers:
        info += f" | 📚 Known: {', '.join(known_speakers)}"
    print(f"   {info}")

def print_db_analytics(analytics: Dict):
    """Print formatted database analytics"""
    print(f"""📊 Database Latency Analytics:
   Total responses: {analytics['total_responses']}
   Interruption rate: {analytics['interruption_rate']:.1%}
   High latency rate: {analytics['high_latency_rate']:.1%}
   Model switch rate: {analytics['model_switch_rate']:.1%}
   Model usage: {analytics['model_usage']}
   Avg response times: {analytics['avg_response_times']}""")

def process_feedback(text: str) -> Dict:
    """Process user feedback text into structured data"""
    text_lower = text.lower().strip()
    feedback = {"helpful": "unknown"}
    
    if text_lower in ['yes', 'y', 'helpful']:
        feedback["helpful"] = True
    elif text_lower in ['no', 'n', 'not helpful']:
        feedback["helpful"] = False
    elif text_lower in ['partially', 'somewhat']:
        feedback["helpful"] = "partial"
    
    return feedback

def handle_special_commands(text: str, gemma_client, conversation_manager) -> bool:
    """Handle special voice commands. Returns True if command was handled."""
    text_lower = text.lower()
    
    if text_lower == "latency status":
        gemma_client.print_latency_status()
        return True
    
    if text_lower == "database analytics":
        if conversation_manager.vector_db:
            analytics = conversation_manager.vector_db.get_latency_analytics()
            if analytics.get("status") == "no_data":
                print("📊 No analytics data available yet")
            else:
                print_db_analytics(analytics)
        else:
            print("📊 Database not available")
        return True
    
    if text_lower == "monitoring status":
        from ai.adaptive_system_monitor import adaptive_monitor
        s = adaptive_monitor.get_status_report()
        active = '✓' if s.get('monitoring_active') else '✗'
        allowed = '✓' if s.get('monitoring_allowed') else '✗'
        print(f"🤖 Monitor: {s.get('system_mode')} | Active: {active} | Allowed: {allowed} | Changes: {s.get('recent_parameter_changes', 0)}")
        return True
    
    if text_lower in ["save config", "save configuration"]:
        from config.runtime_config import runtime_config
        if runtime_config.save_config():
            print("💾 Configuration saved to disk")
        else:
            print("❌ Failed to save configuration")
        return True
    
    if text_lower in ["reset config", "reset configuration", "reset to defaults"]:
        from config.runtime_config import runtime_config
        if runtime_config.reset_to_defaults():
            print("🔄 Configuration reset to defaults")
        else:
            print("❌ Failed to reset configuration")
        return True
    
    return False

def get_audio_error_message(errno: int) -> Optional[str]:
    """Get appropriate error message for audio errors"""
    if errno == -9981:
        return None  # Common buffer error, ignore
    return f"Audio error: {errno}"