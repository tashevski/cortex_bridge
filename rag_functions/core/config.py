"""Configuration for RAG functions with Gemma integration"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration for RAG system with Gemma"""
    
    # Ollama server settings
    ollama_base_url: str = "http://localhost:11434"
    request_timeout: int = 60  # Timeout in seconds for Gemma requests
    
    # RAG settings
    max_reference_chunks: int = 5  # Maximum number of reference chunks to retrieve
    chunk_size: int = 500  # Size of text chunks for vector DB
    include_references: bool = True  # Include reference documents
    
    # Report settings
    detailed_report: bool = True  # Whether to generate detailed analysis
    
    # Prompt template settings
    use_prompt_template: bool = True  # Whether to use structured prompt templates
    default_template: str = "structured_analysis"  # Default template to use
    custom_template: Optional[str] = None  # Custom template override
    
    # Debug settings
    verbose: bool = False  # Print debug information
    save_intermediate_results: bool = True  # Save intermediate processing results

# Default configuration
default_config = RAGConfig()

def get_config() -> RAGConfig:
    """Get default configuration"""
    return default_config