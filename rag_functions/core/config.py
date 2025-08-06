from dataclasses import dataclass

@dataclass
class RAGConfig:
    max_reference_chunks: int = 5
    use_prompt_template: bool = False
    custom_template: str = None

def get_config():
    return RAGConfig()