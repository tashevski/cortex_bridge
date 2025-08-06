#!/usr/bin/env python3
"""Simplified version of main.py for testing"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_functions.core.config import get_config
import re
import json

def process_document_simple(file_path, reference_texts=None, use_medical_templates=True, generate_cue_cards=True, context_type="medical"):
    """Simplified version of process_document that avoids problematic imports"""
    
    config = get_config()
    
    print(f"Processing document: {file_path}")
    print(f"Config loaded: {config}")
    
    # For now, just return a simple response
    return {
        'analysis': f"Simplified analysis of {file_path}",
        'cue_cards': {'main': {'cue_cards': ['Test cue card 1', 'Test cue card 2']}},
        'template_info': None,
        'document_type_scores': None
    }

if __name__ == "__main__":
    result = process_document_simple("/Users/alexander/Library/CloudStorage/Dropbox/Personal Research/cortex_bridge/paper/bsp_2.pdf")
    print("Result:", json.dumps(result, indent=2)) 