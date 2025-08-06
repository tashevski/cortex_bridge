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
        'exit', 'quit', 'stop', 'bye', 'goodbye', 'end conversation', 'shut up'
    ])
    
    # Contextual exit detection - help phrases from LLM that should trigger exit on negative response
    help_phrases: List[str] = field(default_factory=lambda: [
        'do you need help with anything else',
        'is there anything else i can help you with',
        'can i help you with anything else',
        'anything else you need',
        'do you have any other questions',
        'is there anything else',
        'anything else i can assist with',
        'do you need anything else',
        'can i assist with anything else'
    ])
    
    # Negative responses that should trigger exit when following help phrases
    negative_responses: List[str] = field(default_factory=lambda: [
        'no', 'nope', 'not really', 'not right now', 'i\'m good', 
        'i\'m fine', 'that\'s all', 'that\'s it', 'nothing else',
        'no thanks', 'no thank you', 'i don\'t think so', 'not at the moment'
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
    
    # Emotion-based triggering
    enter_on_emotions: bool = False  # Whether emotions can trigger Gemma mode
    trigger_emotions: List[str] = field(default_factory=lambda: [
        'anger', 'frustration', 'sadness', 'fear', 'surprise'
    ])  # Emotions that trigger entry
    emotion_confidence_threshold: float = 0.7  # Minimum confidence for emotion trigger
    emotion_window_size: int = 5  # Number of recent inputs to analyze
    emotion_trigger_count: int = 2  # How many emotion instances needed in window
    
    # Conversation context formatting
    max_context_messages: int = 6  # Max messages in context
    max_history_items: int = 100  # Max items to keep in history
    use_vector_context = True  # Enable vector context in responses

@dataclass
class SpeechProcessorConfig:
    """Configuration for SpeechProcessor (Voice Activity Detection)"""
    sample_rate: int = 16000  # Audio sample rate
    vad_aggressiveness: int = 2  # WebRTC VAD aggressiveness (0-3, higher = more aggressive)
    silence_threshold: int = 3  # Frames of silence before marking as not speaking
    energy_threshold: float = 500.0  # Fallback energy threshold for speech detection
    frame_size: int = 960  # Audio frame size for VAD processing

@dataclass
class SpeakerDetectorConfig:
    """Configuration for SpeakerDetector"""
    # Core detection parameters
    max_speakers: int = 8  # Maximum number of speakers to track
    buffer_size: int = 16000  # Audio buffer size (1 second at 16kHz)
    similarity_threshold: float = 0.40  # Threshold for speaker matching
    min_speech_energy: float = 0.02  # Minimum energy to consider as speech
    
    # Speaker change detection
    min_frames_for_new_speaker: int = 15  # Frames needed to register new speaker
    min_frames_for_change: int = 4  # Frames needed to confirm speaker change
    
    # Model settings
    use_ecapa_model: bool = True  # Use ECAPA-TDNN model (fallback to spectral if False)
    model_save_dir: str = "models/spkrec-ecapa-voxceleb"  # Directory for speaker model
    
    # Embedding parameters
    embedding_alpha: float = 0.05  # Moving average factor for embedding updates
    normalize_embeddings: bool = True  # L2 normalize embeddings
    
    # Spectral fallback parameters (when ECAPA not available)
    fft_size: int = 1024  # FFT size for spectral analysis
    voice_freq_min: float = 80.0  # Minimum voice frequency (Hz)
    voice_freq_max: float = 4000.0  # Maximum voice frequency (Hz)
    spectral_bands: int = 8  # Number of frequency bands for features

@dataclass
class VoskModelConfig:
    """Configuration for Vosk speech recognition model"""
    # Model preference (in order of preference)
    preferred_models: List[str] = field(default_factory=lambda: ["large", "medium", "small"])
    
    # Available models
    available_models: dict = field(default_factory=lambda: {
        "small": {"name": "vosk-model-small-en-us-0.15", "accuracy": "basic"},
        "medium": {"name": "vosk-model-en-us-0.22", "accuracy": "good"},
        "large": {"name": "vosk-model-en-us-0.21", "accuracy": "high"}
    })
    
    # https://alphacephei.com/vosk/models/vosk-model-en-us-0.21.zip
    # Model directory
    models_base_dir: str = "models"  # Base directory for model storage
    
    # Recognition parameters
    sample_rate: int = 16000  # Audio sample rate for recognition
    
    # Fallback model if none found
    fallback_model_name: str = "vosk-model-en-us-0.22"

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
    speech_processor: SpeechProcessorConfig = field(default_factory=SpeechProcessorConfig)
    speaker_detector: SpeakerDetectorConfig = field(default_factory=SpeakerDetectorConfig)
    vosk_model: VoskModelConfig = field(default_factory=VoskModelConfig)

cfg = Config()
