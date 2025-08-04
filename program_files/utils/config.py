from dataclasses import dataclass

@dataclass
class Config:
    gemma_client_context_length_threshold: int = 500
    gemma_client_complex_keywords: list = ['analyze', 'explain', 'reasoning', 'complex', 'detailed', 'comprehensive']
    gemma_client_simple_keywords: list = ['what', 'when', 'where', 'yes', 'no', 'quick', 'simple']
    gemma_client_default_model: str = "gemma3n:e2b"
    gemma_client_base_url: str = "http://localhost:11434"
    gemma_client_switch_threshold: int = 30
    gemma_client_latency_monitor_threshold: int = 10
    gemma_client_latency_monitor_window: int = 10
    gemma_client_latency_monitor_window_size: int = 10
    

cfg = Config()
