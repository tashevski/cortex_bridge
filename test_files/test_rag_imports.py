#!/usr/bin/env python3
"""Test script to verify rag_functions imports are working"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("Testing rag_functions imports...")

# Test basic imports
try:
    from rag_functions.utils.retrieval import setup_vector_db, retrieve_references, extract_medical_issues_list
    print("✓ Successfully imported retrieval functions")
except ImportError as e:
    print(f"✗ Failed to import retrieval: {e}")

try:
    from rag_functions.utils.semantic_parser import parse_document
    print("✓ Successfully imported semantic_parser")
except ImportError as e:
    print(f"✗ Failed to import semantic_parser: {e}")

try:
    from rag_functions.core.config import get_config, RAGConfig
    print("✓ Successfully imported config")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")

try:
    from rag_functions.core.llm_analysis import analyze_with_llm, create_cue_cards
    print("✓ Successfully imported llm_analysis")
except ImportError as e:
    print(f"✗ Failed to import llm_analysis: {e}")

try:
    from rag_functions.templates.prompt_templates import get_template
    print("✓ Successfully imported prompt_templates")
except ImportError as e:
    print(f"✗ Failed to import prompt_templates: {e}")

try:
    from program_files.ai.gemma_client import GemmaClient
    print("✓ Successfully imported GemmaClient")
except ImportError as e:
    print(f"✗ Failed to import GemmaClient: {e}")

# Test that we can create basic objects
try:
    config = get_config()
    print(f"✓ Successfully created config: {config}")
except Exception as e:
    print(f"✗ Failed to create config: {e}")

print("\nImport test complete!")
print("\nNote: Some modules like ocr_layout_copy and ML modules require additional")
print("dependencies (layoutparser, sentence_transformers) which may not be available.")