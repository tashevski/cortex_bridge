"""Adaptive system monitor that optimizes parameters based on program performance"""
from typing import Dict, Any, Optional, List, Tuple
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from config.runtime_config import runtime_config
from config.config import cfg
# Lazy import to avoid initialization delays
# from database.enhanced_conversation_db import EnhancedConversationDB
import logging
import statistics
import threading

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """System operational modes"""
    LISTENING = "listening"      # System is listening for speech input
    GEMMA = "gemma"             # LLM is actively processing/responding
    PROCESSING = "processing"    # Other processing (speaker detection, etc.)
    IDLE = "idle"               # System is idle
    SHUTDOWN = "shutdown"       # System is shutting down

@dataclass
class SystemMetrics:
    """System performance metrics collected from database"""
    avg_response_time: float
    error_rate: float
    speaker_change_frequency: float
    conversation_length_avg: float
    model_switch_frequency: float
    interruption_rate: float
    successful_responses: int
    failed_responses: int
    timestamp: datetime
    
@dataclass
class PerformanceAnalysis:
    """Analysis of system performance"""
    response_time_trend: str  # 'improving', 'degrading', 'stable'
    error_trend: str
    speaker_detection_accuracy: float
    model_efficiency: float
    user_satisfaction_proxy: float  # Based on conversation flow
    bottlenecks: List[str]  # Identified performance bottlenecks

class AdaptiveSystemMonitor:
    """System monitor that optimizes parameters based on performance metrics"""
    
    def __init__(self, db_path: Optional[str] = None):
        # Lazy initialize database to avoid startup delays
        self.db_path = db_path
        self._db = None
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = 30  # seconds between checks
        self.metrics_history = []
        self.max_history = 100
        
        # System mode tracking
        self.current_mode = SystemMode.IDLE
        self.mode_lock = threading.Lock()
        self.last_mode_change = time.time()
        self.mode_history = []  # Track mode transitions
        
        # Performance thresholds
        self.performance_thresholds = {
            'response_time_slow': 3.0,  # seconds
            'response_time_fast': 1.0,
            'error_rate_high': 0.1,    # 10%
            'speaker_change_frequent': 0.5,  # changes per minute
            'interruption_rate_high': 0.3,   # 30%
        }
        
        # Track parameter change history to avoid oscillation
        self.recent_changes = {}
        self.change_cooldown = 60  # seconds before allowing same parameter change again
        
        # Modes where monitoring/optimization should be paused
        self.passive_modes = {SystemMode.GEMMA, SystemMode.SHUTDOWN}
    
    @property
    def db(self):
        """Lazy-loaded database connection"""
        if self._db is None:
            from database.enhanced_conversation_db import EnhancedConversationDB
            self._db = EnhancedConversationDB(self.db_path) if self.db_path else EnhancedConversationDB()
        return self._db
    
    def set_system_mode(self, mode: SystemMode, context: Optional[str] = None):
        """Set the current system mode"""
        with self.mode_lock:
            if self.current_mode != mode:
                old_mode = self.current_mode
                self.current_mode = mode
                self.last_mode_change = time.time()
                
                # Record mode transition
                self.mode_history.append({
                    'from_mode': old_mode.value,
                    'to_mode': mode.value,
                    'timestamp': datetime.now(),
                    'context': context
                })
                
                # Keep mode history manageable
                if len(self.mode_history) > 100:
                    self.mode_history.pop(0)
                
                logger.info(f"System mode changed: {old_mode.value} → {mode.value}" + 
                           (f" ({context})" if context else ""))
                
                # Handle mode-specific actions
                self._handle_mode_transition(old_mode, mode)
    
    def get_system_mode(self) -> SystemMode:
        """Get the current system mode"""
        with self.mode_lock:
            return self.current_mode
    
    def is_monitoring_allowed(self) -> bool:
        """Check if monitoring/optimization is allowed in current mode"""
        with self.mode_lock:
            return self.current_mode not in self.passive_modes
    
    def _handle_mode_transition(self, old_mode: SystemMode, new_mode: SystemMode):
        """Handle actions needed when transitioning between modes"""
        # Entering Gemma mode - pause intensive operations
        if new_mode == SystemMode.GEMMA:
            logger.debug("Entering Gemma mode - pausing optimization activities")
            # Could potentially lower priority of monitoring thread here
        
        # Exiting Gemma mode - resume normal operations
        elif old_mode == SystemMode.GEMMA and new_mode != SystemMode.SHUTDOWN:
            logger.debug("Exiting Gemma mode - resuming optimization activities")
            # Could trigger immediate metrics collection to assess Gemma performance
        
        # Entering shutdown mode - prepare for cleanup
        elif new_mode == SystemMode.SHUTDOWN:
            logger.info("System entering shutdown mode")
    
    def collect_system_metrics(self, time_window_minutes: int = 10) -> SystemMetrics:
        """Collect system performance metrics from database"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        try:
            # Get recent conversations
            conversations = self.db.get_conversations_by_date_range(start_time, end_time)
            
            if not conversations:
                return self._create_empty_metrics()
            
            # Calculate metrics
            response_times = []
            errors = 0
            total_responses = 0
            speaker_changes = 0
            interruptions = 0
            conversation_lengths = []
            
            for conv in conversations:
                # Get conversation details
                messages = self.db.get_conversation_history(conv['id'])
                if not messages:
                    continue
                    
                total_responses += len(messages)
                conversation_lengths.append(len(messages))
                
                # Analyze response times and errors
                for msg in messages:
                    if 'response_time' in msg:
                        response_times.append(msg['response_time'])
                    if msg.get('error', False):
                        errors += 1
                    if msg.get('speaker_changed', False):
                        speaker_changes += 1
                    if msg.get('interrupted', False):
                        interruptions += 1
            
            # Calculate averages and rates
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            error_rate = errors / max(total_responses, 1)
            interruption_rate = interruptions / max(total_responses, 1)
            speaker_change_freq = speaker_changes / max(time_window_minutes, 1)
            conversation_length_avg = statistics.mean(conversation_lengths) if conversation_lengths else 0.0
            
            return SystemMetrics(
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                speaker_change_frequency=speaker_change_freq,
                conversation_length_avg=conversation_length_avg,
                model_switch_frequency=self._calculate_model_switch_frequency(conversations),
                interruption_rate=interruption_rate,
                successful_responses=total_responses - errors,
                failed_responses=errors,
                timestamp=end_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return self._create_empty_metrics()
    
    def analyze_performance(self, metrics: SystemMetrics) -> PerformanceAnalysis:
        """Analyze current performance against historical data"""
        # Add current metrics to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        if len(self.metrics_history) < 2:
            # Not enough data for trend analysis
            return PerformanceAnalysis(
                response_time_trend='stable',
                error_trend='stable',
                speaker_detection_accuracy=0.8,  # Default assumption
                model_efficiency=0.7,
                user_satisfaction_proxy=0.6,
                bottlenecks=[]
            )
        
        # Analyze trends
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        older_metrics = self.metrics_history[-10:-5] if len(self.metrics_history) >= 10 else self.metrics_history[:-5]
        
        # Response time trend
        recent_avg_response = statistics.mean([m.avg_response_time for m in recent_metrics])
        older_avg_response = statistics.mean([m.avg_response_time for m in older_metrics]) if older_metrics else recent_avg_response
        
        response_time_trend = self._determine_trend(recent_avg_response, older_avg_response)
        
        # Error rate trend
        recent_error_rate = statistics.mean([m.error_rate for m in recent_metrics])
        older_error_rate = statistics.mean([m.error_rate for m in older_metrics]) if older_metrics else recent_error_rate
        
        error_trend = self._determine_trend(recent_error_rate, older_error_rate)
        
        # Calculate efficiency metrics
        speaker_accuracy = self._calculate_speaker_accuracy(recent_metrics)
        model_efficiency = self._calculate_model_efficiency(recent_metrics)
        user_satisfaction = self._calculate_user_satisfaction_proxy(recent_metrics)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(metrics)
        
        return PerformanceAnalysis(
            response_time_trend=response_time_trend,
            error_trend=error_trend,
            speaker_detection_accuracy=speaker_accuracy,
            model_efficiency=model_efficiency,
            user_satisfaction_proxy=user_satisfaction,
            bottlenecks=bottlenecks
        )
    
    def optimize_parameters(self, metrics: SystemMetrics, analysis: PerformanceAnalysis) -> Dict[str, Any]:
        """Optimize system parameters based on performance analysis"""
        optimizations = {}
        
        # Response time optimizations
        if metrics.avg_response_time > self.performance_thresholds['response_time_slow']:
            if self._can_change_parameter('latency_monitor.high_latency_threshold'):
                changes = runtime_config.update_config('latency_monitor',
                    high_latency_threshold=max(1.5, metrics.avg_response_time * 0.7),
                    emergency_switch_threshold=0.4  # More aggressive switching
                )
                optimizations['latency_monitor'] = changes
                self._record_parameter_change('latency_monitor.high_latency_threshold')
                logger.info(f"Lowered latency threshold due to slow responses: {metrics.avg_response_time:.2f}s")
        
        elif metrics.avg_response_time < self.performance_thresholds['response_time_fast']:
            if self._can_change_parameter('latency_monitor.high_latency_threshold'):
                changes = runtime_config.update_config('latency_monitor',
                    high_latency_threshold=min(5.0, metrics.avg_response_time * 1.5),
                    emergency_switch_threshold=0.6  # Less aggressive switching
                )
                optimizations['latency_monitor'] = changes
                self._record_parameter_change('latency_monitor.high_latency_threshold')
                logger.info(f"Raised latency threshold due to fast responses: {metrics.avg_response_time:.2f}s")
        
        # Error rate optimizations
        if metrics.error_rate > self.performance_thresholds['error_rate_high']:
            if self._can_change_parameter('smart_model_selector.switch_threshold'):
                changes = runtime_config.update_config('smart_model_selector',
                    switch_threshold=max(15, cfg.smart_model_selector.switch_threshold - 10),
                    context_length_threshold=max(300, cfg.smart_model_selector.context_length_threshold - 100)
                )
                optimizations['smart_model_selector'] = changes
                self._record_parameter_change('smart_model_selector.switch_threshold')
                logger.info(f"Adjusted model selection due to high error rate: {metrics.error_rate:.2%}")
        
        # Speaker detection optimizations
        if metrics.speaker_change_frequency > self.performance_thresholds['speaker_change_frequent']:
            if self._can_change_parameter('speaker_detector.similarity_threshold'):
                changes = runtime_config.update_config('speaker_detector',
                    similarity_threshold=min(0.7, cfg.speaker_detector.similarity_threshold + 0.05),
                    min_frames_for_change=min(10, cfg.speaker_detector.min_frames_for_change + 1)
                )
                optimizations['speaker_detector'] = changes
                self._record_parameter_change('speaker_detector.similarity_threshold')
                logger.info(f"Increased speaker detection threshold due to frequent changes: {metrics.speaker_change_frequency:.2f}/min")
        
        # Interruption rate optimizations
        if metrics.interruption_rate > self.performance_thresholds['interruption_rate_high']:
            if self._can_change_parameter('speech_processor.silence_threshold'):
                changes = runtime_config.update_config('speech_processor',
                    silence_threshold=min(5, cfg.speech_processor.silence_threshold + 1),
                    energy_threshold=min(1000, cfg.speech_processor.energy_threshold + 100)
                )
                optimizations['speech_processor'] = changes
                self._record_parameter_change('speech_processor.silence_threshold')
                logger.info(f"Adjusted speech detection due to high interruption rate: {metrics.interruption_rate:.2%}")
        
        return optimizations
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started adaptive system monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped adaptive system monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Check if monitoring is allowed in current mode
                if not self.is_monitoring_allowed():
                    logger.debug(f"Monitoring paused - system in {self.current_mode.value} mode")
                    time.sleep(self.check_interval)
                    continue
                
                # Collect metrics
                metrics = self.collect_system_metrics()
                
                # Only analyze and optimize if not in passive mode
                if self.is_monitoring_allowed():
                    # Analyze performance
                    analysis = self.analyze_performance(metrics)
                    
                    # Optimize parameters
                    optimizations = self.optimize_parameters(metrics, analysis)
                    
                    if optimizations:
                        logger.info(f"Applied {len(optimizations)} parameter optimizations")
                else:
                    logger.debug("Skipping optimization - system in passive mode")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _create_empty_metrics(self) -> SystemMetrics:
        """Create empty metrics when no data available"""
        return SystemMetrics(
            avg_response_time=0.0,
            error_rate=0.0,
            speaker_change_frequency=0.0,
            conversation_length_avg=0.0,
            model_switch_frequency=0.0,
            interruption_rate=0.0,
            successful_responses=0,
            failed_responses=0,
            timestamp=datetime.now()
        )
    
    def _calculate_model_switch_frequency(self, conversations: List[Dict]) -> float:
        """Calculate how often models are being switched"""
        switches = 0
        total_messages = 0
        
        for conv in conversations:
            messages = self.db.get_conversation_history(conv['id'])
            if len(messages) > 1:
                for i in range(1, len(messages)):
                    if messages[i].get('model') != messages[i-1].get('model'):
                        switches += 1
                total_messages += len(messages)
        
        return switches / max(total_messages, 1) if total_messages > 0 else 0.0
    
    def _determine_trend(self, recent: float, older: float) -> str:
        """Determine if metric is improving, degrading, or stable"""
        if abs(recent - older) < 0.1:  # 10% threshold
            return 'stable'
        elif recent > older:
            return 'degrading'
        else:
            return 'improving'
    
    def _calculate_speaker_accuracy(self, metrics_list: List[SystemMetrics]) -> float:
        """Estimate speaker detection accuracy based on change frequency"""
        avg_change_freq = statistics.mean([m.speaker_change_frequency for m in metrics_list])
        # Assume optimal change frequency is around 0.1-0.3 per minute
        if 0.1 <= avg_change_freq <= 0.3:
            return 0.9
        elif avg_change_freq < 0.1:
            return 0.7  # Possibly missing changes
        else:
            return max(0.3, 1.0 - (avg_change_freq - 0.3) * 0.5)  # Too many changes
    
    def _calculate_model_efficiency(self, metrics_list: List[SystemMetrics]) -> float:
        """Calculate model switching efficiency"""
        avg_switch_freq = statistics.mean([m.model_switch_frequency for m in metrics_list])
        avg_response_time = statistics.mean([m.avg_response_time for m in metrics_list])
        
        # Efficient switching should balance performance with stability
        efficiency = 1.0 - min(0.5, avg_switch_freq)  # Penalize excessive switching
        efficiency *= max(0.5, 1.0 - (avg_response_time / 5.0))  # Reward fast responses
        
        return max(0.1, efficiency)
    
    def _calculate_user_satisfaction_proxy(self, metrics_list: List[SystemMetrics]) -> float:
        """Estimate user satisfaction based on conversation flow"""
        avg_error_rate = statistics.mean([m.error_rate for m in metrics_list])
        avg_interruption_rate = statistics.mean([m.interruption_rate for m in metrics_list])
        avg_conv_length = statistics.mean([m.conversation_length_avg for m in metrics_list])
        
        # Higher satisfaction with fewer errors, fewer interruptions, longer conversations
        satisfaction = 1.0 - avg_error_rate * 2  # Errors heavily impact satisfaction
        satisfaction -= avg_interruption_rate * 0.5  # Interruptions are annoying
        satisfaction += min(0.2, avg_conv_length / 50)  # Longer conversations suggest engagement
        
        return max(0.0, min(1.0, satisfaction))
    
    def _identify_bottlenecks(self, metrics: SystemMetrics) -> List[str]:
        """Identify system bottlenecks based on metrics"""
        bottlenecks = []
        
        if metrics.avg_response_time > self.performance_thresholds['response_time_slow']:
            bottlenecks.append('slow_response_time')
        
        if metrics.error_rate > self.performance_thresholds['error_rate_high']:
            bottlenecks.append('high_error_rate')
        
        if metrics.speaker_change_frequency > self.performance_thresholds['speaker_change_frequent']:
            bottlenecks.append('excessive_speaker_changes')
        
        if metrics.interruption_rate > self.performance_thresholds['interruption_rate_high']:
            bottlenecks.append('high_interruption_rate')
        
        if metrics.model_switch_frequency > 0.5:  # More than 50% of messages trigger switches
            bottlenecks.append('excessive_model_switching')
        
        return bottlenecks
    
    def _can_change_parameter(self, param_key: str) -> bool:
        """Check if parameter can be changed (cooldown check)"""
        last_change = self.recent_changes.get(param_key, 0)
        return time.time() - last_change > self.change_cooldown
    
    def _record_parameter_change(self, param_key: str):
        """Record when a parameter was changed"""
        self.recent_changes[param_key] = time.time()
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get current system status report"""
        with self.mode_lock:
            current_mode = self.current_mode.value
            mode_duration = time.time() - self.last_mode_change
            monitoring_allowed = self.is_monitoring_allowed()
        
        if not self.metrics_history:
            return {
                "status": "no_data", 
                "message": "No metrics collected yet",
                "system_mode": current_mode,
                "mode_duration_seconds": mode_duration,
                "monitoring_allowed": monitoring_allowed,
                "monitoring_active": self.monitoring
            }
        
        latest_metrics = self.metrics_history[-1]
        analysis = self.analyze_performance(latest_metrics)
        
        # Get recent mode transitions
        recent_transitions = self.mode_history[-5:] if self.mode_history else []
        
        return {
            "status": "active",
            "timestamp": latest_metrics.timestamp.isoformat(),
            "system_mode": current_mode,
            "mode_duration_seconds": mode_duration,
            "monitoring_allowed": monitoring_allowed,
            "monitoring_active": self.monitoring,
            "recent_mode_transitions": [
                {
                    "from": t["from_mode"],
                    "to": t["to_mode"], 
                    "timestamp": t["timestamp"].isoformat(),
                    "context": t.get("context")
                } for t in recent_transitions
            ],
            "metrics": {
                "avg_response_time": latest_metrics.avg_response_time,
                "error_rate": latest_metrics.error_rate,
                "speaker_change_frequency": latest_metrics.speaker_change_frequency,
                "interruption_rate": latest_metrics.interruption_rate,
            },
            "analysis": {
                "response_time_trend": analysis.response_time_trend,
                "error_trend": analysis.error_trend,
                "speaker_detection_accuracy": analysis.speaker_detection_accuracy,
                "model_efficiency": analysis.model_efficiency,
                "user_satisfaction_proxy": analysis.user_satisfaction_proxy,
                "bottlenecks": analysis.bottlenecks,
            },
            "recent_parameter_changes": len(self.recent_changes),
        }

# Global instance for easy access
adaptive_monitor = AdaptiveSystemMonitor()