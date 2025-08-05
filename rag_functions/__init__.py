# __init__.py for rag_functions

# Core functionality
from .core import (
    process_document, analyze_with_llm, process_with_gemma,
    RAGConfig, get_config, process_medical_document
)

# Template functionality  
from .templates import (
    get_template, list_templates, get_template_for_use_case,
    create_custom_template, PromptTemplate
)

# ML and AI functionality
from .ml import (
    select_optimal_templates, analyze_document_type,
    extract_cue_cards, CueCard, format_cue_cards
)

# Utility functions
from .utils import (
    setup_vector_db, retrieve_references, 
    parse_document, extract_text_and_layout
)

__all__ = [
    # Core
    'process_document',
    'analyze_with_llm', 
    'process_with_gemma',
    'RAGConfig',
    'get_config',
    'process_medical_document',
    
    # Templates
    'get_template',
    'list_templates', 
    'get_template_for_use_case',
    'create_custom_template',
    'PromptTemplate',
    
    # ML/AI
    'select_optimal_templates',
    'analyze_document_type',
    'extract_cue_cards',
    'CueCard',
    'format_cue_cards',
    
    # Utils
    'setup_vector_db',
    'retrieve_references',
    'parse_document',
    'extract_text_and_layout'
]