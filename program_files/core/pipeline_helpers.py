#!/usr/bin/env python3
"""Helper functions for program_pipeline.py to reduce complexity"""

from typing import Dict, Optional
import json

def handle_gemma_response(gemma_client, text: str, context: str, conversation_manager):
    """Generate and handle Gemma response with latency tracking"""
    response = gemma_client.generate_response_optimized(text, context)
    if response:
        print(f"🤖 Gemma: {response}")
        latency_metrics = gemma_client.get_last_latency_metrics()
        conversation_manager.add_to_history(response, False, "Gemma", latency_metrics=latency_metrics)
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