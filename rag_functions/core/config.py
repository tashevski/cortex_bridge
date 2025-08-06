from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    max_reference_chunks: int = 5
    use_prompt_template: bool = False
    custom_template: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    verbose: bool = False
    request_timeout: int = 120
    default_template: str = "medical_analysis"

def get_config():
    return RAGConfig()