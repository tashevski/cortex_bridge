#!/usr/bin/env python3
"""Latency monitoring and adaptive response system"""

import time
import threading
from collections import deque
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from config.config import LatencyMonitorConfig

@dataclass
class LatencyMetrics:
    """Metrics for tracking response latency and user behavior"""
    response_time: float
    user_spoke_during_response: bool
    speech_activity_during_response: float  # seconds of speech during response
    model_used: str
    context_length: int
    had_image: bool
    timestamp: float

class LatencyMonitor:
    """Monitors response latency and user speech patterns to optimize model selection"""
    
    def __init__(self, config: Optional[LatencyMonitorConfig] = None):
        if config is None:
            from config.config import cfg
            config = cfg.latency_monitor
            
        self.history_size = config.history_size
        self.metrics_history = deque(maxlen=config.history_size)
        self.current_response_start = None
        self.speech_during_response = 0.0
        self.speech_start_time = None
        self.is_monitoring = False
        self.lock = threading.Lock()
        
        # Adaptive thresholds from config
        self.high_latency_threshold = config.high_latency_threshold
        self.acceptable_interruption_rate = config.acceptable_interruption_rate
        self.emergency_switch_threshold = config.emergency_switch_threshold
        
    def start_response_timing(self, model: str, context_length: int, has_image: bool):
        """Start timing a model response"""
        with self.lock:
            self.current_response_start = time.time()
            self.speech_during_response = 0.0
            self.is_monitoring = True
            self.current_model = model
            self.current_context_length = context_length
            self.current_has_image = has_image
    
    def record_speech_activity(self, is_speech: bool):
        """Record when user is speaking during model response"""
        if not self.is_monitoring:
            return
            
        with self.lock:
            current_time = time.time()
            
            if is_speech and self.speech_start_time is None:
                # User started speaking
                self.speech_start_time = current_time
            elif not is_speech and self.speech_start_time is not None:
                # User stopped speaking
                speech_duration = current_time - self.speech_start_time
                self.speech_during_response += speech_duration
                self.speech_start_time = None
    
    def end_response_timing(self) -> LatencyMetrics:
        """End timing and record metrics"""
        with self.lock:
            if not self.is_monitoring or self.current_response_start is None:
                return None
                
            end_time = time.time()
            response_time = end_time - self.current_response_start
            
            # Handle case where user is still speaking when response ends
            if self.speech_start_time is not None:
                speech_duration = end_time - self.speech_start_time
                self.speech_during_response += speech_duration
                self.speech_start_time = None
            
            metrics = LatencyMetrics(
                response_time=response_time,
                user_spoke_during_response=self.speech_during_response > 0.5,  # >0.5s = interruption
                speech_activity_during_response=self.speech_during_response,
                model_used=self.current_model,
                context_length=self.current_context_length,
                had_image=self.current_has_image,
                timestamp=end_time
            )
            
            self.metrics_history.append(metrics)
            self.is_monitoring = False
            
            return metrics
    
    def get_interruption_rate(self, recent_count: int = 10) -> float:
        """Get the rate of user interruptions in recent responses"""
        if not self.metrics_history:
            return 0.0
            
        recent_metrics = list(self.metrics_history)[-recent_count:]
        interruptions = sum(1 for m in recent_metrics if m.user_spoke_during_response)
        
        return interruptions / len(recent_metrics)
    
    def get_avg_response_time(self, model: str = None, recent_count: int = 20) -> float:
        """Get average response time, optionally filtered by model"""
        if not self.metrics_history:
            return 0.0
            
        recent_metrics = list(self.metrics_history)[-recent_count:]
        
        if model:
            filtered_metrics = [m for m in recent_metrics if m.model_used == model]
        else:
            filtered_metrics = recent_metrics
            
        if not filtered_metrics:
            return 0.0
            
        return sum(m.response_time for m in filtered_metrics) / len(filtered_metrics)
    
    def should_prioritize_speed(self) -> bool:
        """Determine if we should prioritize speed over capability"""
        recent_interruption_rate = self.get_interruption_rate(recent_count=5)
        overall_interruption_rate = self.get_interruption_rate(recent_count=20)
        
        # Emergency speed mode if recent interruptions are high
        if recent_interruption_rate >= self.emergency_switch_threshold:
            return True
            
        # General speed priority if interruptions are consistently above threshold
        if overall_interruption_rate >= self.acceptable_interruption_rate:
            return True
            
        return False
    
    def get_latency_analysis(self) -> Dict[str, Any]:
        """Get latency analysis"""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        return {
            "total_responses": len(self.metrics_history),
            "recent_interruption_rate": self.get_interruption_rate(5),
            "overall_interruption_rate": self.get_interruption_rate(),
            "avg_response_time_e2b": self.get_avg_response_time("gemma3n:e2b"),
            "avg_response_time_e4b": self.get_avg_response_time("gemma3n:e4b"),
            "should_prioritize_speed": self.should_prioritize_speed()
        }
    
    def get_model_recommendation(self, default_recommendation: str) -> tuple[str, str]:
        """Get model recommendation based on latency patterns"""
        if self.should_prioritize_speed():
            if default_recommendation == "gemma3n:e4b":
                return "gemma3n:e2b", "🚨 Switching to faster model due to user interruptions"
            else:
                return default_recommendation, "⚡ Staying with fast model due to latency concerns"
        
        return default_recommendation, "✅ Using recommended model"
    
    def print_status(self):
        """Print current latency status"""
        analysis = self.get_latency_analysis()
        
        if analysis.get("status") == "no_data":
            print("📊 Latency Monitor: No data yet")
            return
            
        print(f"""📊 Latency Monitor Status:
   Recent interruption rate: {analysis['recent_interruption_rate']:.1%}
   Overall interruption rate: {analysis['overall_interruption_rate']:.1%}
   Avg response time e2b: {analysis['avg_response_time_e2b']:.2f}s
   Avg response time e4b: {analysis['avg_response_time_e4b']:.2f}s
   Speed priority mode: {'ON' if analysis['should_prioritize_speed'] else 'OFF'}
   Recent high latency: {analysis['recent_high_latency_count']}/10 responses""")