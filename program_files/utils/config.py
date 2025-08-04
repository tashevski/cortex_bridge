from dataclasses import dataclass, field
from typing import List

@dataclass
class SmartModelSelectorConfig:
    """Configuration for SmartModelSelector"""
    switch_threshold: int = 30  # seconds before considering switch
    context_length_threshold: int = 500  # characters to trigger e4b
    complex_keywords: List[str] = field(default_factory=lambda: [
        'analyze', 'explain', 'reasoning', 'complex', 'detailed', 'comprehensive'
    ])
    simple_keywords: List[str] = field(default_factory=lambda: [
        'what', 'when', 'where', 'yes', 'no', 'quick', 'simple'
    ])

@dataclass
class LatencyMonitorConfig:
    """Configuration for LatencyMonitor"""
    history_size: int = 50
    high_latency_threshold: float = 3.0  # seconds
    acceptable_interruption_rate: float = 0.2  # 20%
    emergency_switch_threshold: float = 0.5  # 50%
    recent_count_for_interruption_rate: int = 10  # Recent responses to check
    recent_count_for_avg_response_time: int = 20  # Recent responses for avg calculation

@dataclass
class ModelPreloaderConfig:
    """Configuration for ModelPreloader"""
    base_url: str = "http://localhost:11434"
    timeout: int = 30  # seconds for warming requests
    max_retries: int = 3
    background_rotation_interval: int = 300  # seconds between model rotations
    pull_timeout: int = 300  # seconds for model pulling
    
@dataclass
class ConversationModeConfig:
    """Configuration for conversation mode transitions"""
    # Keywords that trigger entering Gemma mode
    enter_keywords: List[str] = field(default_factory=lambda: [
        'hey gemma', 'gemma', 'assistant', 'help'
    ])
    
    # Keywords that trigger exiting Gemma mode
    exit_keywords: List[str] = field(default_factory=lambda: [
        'exit', 'quit', 'stop', 'bye', 'goodbye', 'end conversation'
    ])
    
    # Question detection parameters
    question_words: List[str] = field(default_factory=lambda: [
        'what', 'how', 'why', 'when', 'where', 'who', 'which'
    ])
    
    auxiliary_prefixes: List[str] = field(default_factory=lambda: [
        'is ', 'are ', 'do ', 'does ', 'can ', 'will '
    ])
    
    # Behavior settings
    enter_on_questions: bool = True  # Whether questions automatically enter Gemma mode
    
    # Conversation context formatting
    max_context_messages: int = 6  # Max messages in context
    max_history_items: int = 100  # Max items to keep in history

@dataclass
class GemmaClientConfig:
    """Configuration for GemmaClient"""
    default_model: str = "gemma3n:e2b"
    base_url: str = "http://localhost:11434"
    timeout: int = 30  # Default timeout for requests
    stream: bool = False  # Stream responses or not

@dataclass
class Config:
    """Main configuration class"""
    smart_model_selector: SmartModelSelectorConfig = field(default_factory=SmartModelSelectorConfig)
    latency_monitor: LatencyMonitorConfig = field(default_factory=LatencyMonitorConfig)
    model_preloader: ModelPreloaderConfig = field(default_factory=ModelPreloaderConfig)
    conversation_mode: ConversationModeConfig = field(default_factory=ConversationModeConfig)
    gemma_client: GemmaClientConfig = field(default_factory=GemmaClientConfig)

cfg = Config()
