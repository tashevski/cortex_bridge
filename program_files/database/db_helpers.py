#!/usr/bin/env python3
"""Helper functions for enhanced_conversation_db.py to reduce complexity"""

from typing import Dict, List, Any
from datetime import datetime

def create_conversation_id(session_id: str) -> str:
    """Generate unique conversation ID"""
    return f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

def create_metadata(session_id: str, speaker: str, role: str, 
                   has_audio: bool, **kwargs) -> Dict:
    """Create metadata dictionary with optional fields"""
    metadata = {
        'session_id': session_id,
        'speaker': speaker,
        'role': role,
        'timestamp': datetime.now().isoformat(),
        'has_audio_features': has_audio
    }
    
    # Add optional fields if provided
    optional_fields = [
        ('emotion_text', 'emotion'),
        ('confidence', 'emotion_confidence'),
        ('feedback', 'feedback_helpful')
    ]
    
    for key, meta_key in optional_fields:
        if key in kwargs and kwargs[key] is not None:
            if key == 'feedback':
                metadata[meta_key] = str(kwargs[key].get('helpful', ''))
            else:
                metadata[meta_key] = kwargs[key]
    
    # Add latency metrics if provided
    if 'latency_metrics' in kwargs and kwargs['latency_metrics']:
        add_latency_to_metadata(metadata, kwargs['latency_metrics'])
    
    # Add model information if provided (for cases where latency_metrics isn't available)
    if 'model_used' in kwargs and kwargs['model_used']:
        if 'model_used' not in metadata:  # Don't override if already set by latency metrics
            metadata['model_used'] = kwargs['model_used']
    
    return metadata

def add_latency_to_metadata(metadata: Dict, latency_metrics: Dict):
    """Add latency metrics to metadata dictionary"""
    latency_fields = [
        ('response_time', 0.0),
        ('user_spoke_during_response', False),
        ('speech_activity_during_response', 0.0),
        ('model_used', 'unknown'),
        ('context_length', 0),
        ('had_image', False),
        ('model_switched', False),
        ('switch_reason', '')
    ]
    
    for field, default in latency_fields:
        metadata[field] = latency_metrics.get(field, default)
    
    # Derived field
    metadata['high_latency'] = latency_metrics.get('response_time', 0.0) > 3.0
    metadata['user_interrupted'] = metadata['user_spoke_during_response']

def calculate_analytics(metrics: List[Dict]) -> Dict[str, Any]:
    """Calculate analytics from metrics list"""
    if not metrics:
        return {"status": "no_data"}
    
    total = len(metrics)
    interrupted = sum(1 for m in metrics if m.get('user_interrupted', False))
    high_latency = sum(1 for m in metrics if m.get('high_latency', False))
    switches = sum(1 for m in metrics if m.get('model_switched', False))
    
    # Model usage and response times
    model_usage = {}
    model_times = {}
    
    for m in metrics:
        model = m.get('model_used', 'unknown')
        model_usage[model] = model_usage.get(model, 0) + 1
        
        if model not in model_times:
            model_times[model] = []
        model_times[model].append(float(m.get('response_time', 0)))
    
    avg_times = {
        model: sum(times) / len(times) 
        for model, times in model_times.items()
    }
    
    return {
        "total_responses": total,
        "interruption_rate": interrupted / total,
        "high_latency_rate": high_latency / total,
        "model_switch_rate": switches / total,
        "model_usage": model_usage,
        "avg_response_times": avg_times,
        "total_interrupted": interrupted,
        "total_high_latency": high_latency,
        "total_model_switches": switches
    }

def analyze_session(session_data: List[Dict]) -> Dict[str, Any]:
    """Analyze a single session for problems"""
    total = len(session_data)
    interrupted = sum(1 for m in session_data if m.get('user_interrupted', False))
    high_latency = sum(1 for m in session_data if m.get('high_latency', False))
    
    return {
        "total": total,
        "interruption_rate": interrupted / total if total > 0 else 0,
        "high_latency_count": high_latency,
        "timestamp": session_data[0].get('timestamp', 'unknown') if session_data else 'unknown'
    }