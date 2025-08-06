"""Minimal adaptive system monitor for parameter optimization"""
from typing import Dict, Any, Optional
import time, threading, logging
from datetime import datetime, timedelta
from enum import Enum
from program_files.config.runtime_config import runtime_config

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    LISTENING = "listening"
    GEMMA = "gemma" 
    PROCESSING = "processing"
    IDLE = "idle"
    SHUTDOWN = "shutdown"

class AdaptiveSystemMonitor:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._db = None
        self.current_mode = SystemMode.IDLE
        self.monitoring = False
        self.mode_lock = threading.Lock()
        self.last_mode_change = time.time()
        self.recent_changes = {}
        
    @property
    def db(self):
        if self._db is None:
            from database.enhanced_conversation_db import EnhancedConversationDB
            self._db = EnhancedConversationDB(self.db_path) if self.db_path else EnhancedConversationDB()
        return self._db
    
    def set_system_mode(self, mode: SystemMode, context: str = ""):
        with self.mode_lock:
            if self.current_mode != mode:
                logger.info(f"Mode: {self.current_mode.value} → {mode.value}" + (f" ({context})" if context else ""))
                self.current_mode = mode
                self.last_mode_change = time.time()
    
    def get_system_mode(self) -> SystemMode:
        return self.current_mode
    
    def is_monitoring_allowed(self) -> bool:
        return self.current_mode not in {SystemMode.GEMMA, SystemMode.SHUTDOWN}
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect basic performance metrics from database"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=10)
            conversations = self.db.get_conversations_by_date_range(start_time, end_time)
            
            if not conversations:
                return {"response_time": 0, "error_rate": 0, "interruptions": 0}
                
            response_times, errors, interruptions = [], 0, 0
            for conv in conversations:
                messages = self.db.get_conversation_history(conv['id'])
                for msg in messages:
                    if msg.get('response_time'): response_times.append(msg['response_time'])
                    if msg.get('error'): errors += 1
                    if msg.get('interrupted'): interruptions += 1
            
            total = len([m for conv in conversations for m in self.db.get_conversation_history(conv['id'])])
            return {
                "response_time": sum(response_times) / len(response_times) if response_times else 0,
                "error_rate": errors / max(total, 1),
                "interruptions": interruptions / max(total, 1)
            }
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {"response_time": 0, "error_rate": 0, "interruptions": 0}
    
    def optimize_parameters(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Apply parameter optimizations based on metrics"""
        if not self.is_monitoring_allowed():
            return {}
            
        changes = {}
        
        # Slow responses -> lower latency thresholds
        if metrics["response_time"] > 3.0 and self._can_change("latency"):
            result = runtime_config.update_config('latency_monitor', high_latency_threshold=2.0)
            if result.get('changed'): changes.update(result['changed'])
            self._record_change("latency")
        
        # High errors -> adjust model selection  
        if metrics["error_rate"] > 0.1 and self._can_change("model"):
            result = runtime_config.update_config('smart_model_selector', switch_threshold=20)
            if result.get('changed'): changes.update(result['changed'])
            self._record_change("model")
        
        # High interruptions -> adjust speech detection
        if metrics["interruptions"] > 0.3 and self._can_change("speech"):
            result = runtime_config.update_config('speech_processor', silence_threshold=4)
            if result.get('changed'): changes.update(result['changed'])
            self._record_change("speech")
            
        return changes
    
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            threading.Thread(target=self._monitor_loop, daemon=True).start()
            logger.info("Adaptive monitoring started")
    
    def stop_monitoring(self):
        self.monitoring = False
        logger.info("Adaptive monitoring stopped")
    
    def _monitor_loop(self):
        while self.monitoring:
            if self.is_monitoring_allowed():
                metrics = self.collect_metrics()
                changes = self.optimize_parameters(metrics)
                if changes:
                    logger.info(f"Applied optimizations: {list(changes.keys())}")
            time.sleep(30)
    
    def _can_change(self, param_type: str) -> bool:
        return time.time() - self.recent_changes.get(param_type, 0) > 60
    
    def _record_change(self, param_type: str):
        self.recent_changes[param_type] = time.time()
    
    def get_status_report(self) -> Dict[str, Any]:
        return {
            "system_mode": self.current_mode.value,
            "monitoring_allowed": self.is_monitoring_allowed(),
            "monitoring_active": self.monitoring,
            "mode_duration_seconds": time.time() - self.last_mode_change,
            "recent_parameter_changes": len(self.recent_changes)
        }

# Global instance
adaptive_monitor = AdaptiveSystemMonitor()