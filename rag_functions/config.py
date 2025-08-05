"""Configuration for RAG functions with Gemma integration"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration for RAG system with Gemma models"""
    
    # Gemma model settings
    use_optimized_client: bool = True  # Use OptimizedGemmaClient vs basic GemmaClient
    default_model: str = "gemma3n:e4b"  # Default Gemma model to use
    fallback_model: str = "gemma:2b"  # Fallback model if default fails
    
    # Ollama server settings
    ollama_base_url: str = "http://localhost:11434"
    request_timeout: int = 60  # Timeout in seconds for Gemma requests
    
    # RAG settings
    max_reference_chunks: int = 5  # Maximum number of reference chunks to retrieve
    chunk_size: int = 500  # Size of text chunks for vector DB
    
    # Analysis settings
    include_calculations: bool = True
    include_references: bool = True  # Include reference documents
    detailed_report: bool = True  # Whether to generate detailed analysis
    
    # Prompt template settings
    use_prompt_template: bool = True  # Whether to use structured prompt templates
    default_template: str = "structured_analysis"  # Default template to use
    custom_template: Optional[str] = None  # Custom template override
    
    # Model selection hints for OptimizedGemmaClient
    prefer_fast_models: bool = False  # Prefer faster models over quality
    enable_latency_monitoring: bool = True  # Monitor and adjust based on latency
    
    # Debug settings
    verbose: bool = False  # Print debug information
    save_intermediate_results: bool = True  # Save intermediate processing results

# Default configuration instance
default_config = RAGConfig()

# Preset configurations for different use cases
PRESETS = {
    "fast": RAGConfig(
        use_optimized_client=True,
        default_model="gemma:2b",
        prefer_fast_models=True,
        detailed_report=False,
        request_timeout=30,
        default_template="concise_qa"
    ),
    
    "quality": RAGConfig(
        use_optimized_client=True,
        default_model="gemma3n:e4b",
        prefer_fast_models=False,
        detailed_report=True,
        request_timeout=120,
        default_template="structured_analysis"
    ),
    
    "balanced": RAGConfig(
        use_optimized_client=True,
        default_model="gemma3n:e4b",
        prefer_fast_models=False,
        detailed_report=True,
        request_timeout=60,
        default_template="structured_analysis"
    ),
    
    "debug": RAGConfig(
        verbose=True,
        save_intermediate_results=True,
        use_optimized_client=False,  # Use basic client for debugging
        request_timeout=300
    )
}

def get_config(preset: str = "balanced") -> RAGConfig:
    """Get configuration by preset name"""
    return PRESETS.get(preset, default_config)