#!/usr/bin/env python3
"""Simple test script to check rag_functions module import"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing basic rag_functions import...")
    import rag_functions
    print("✓ rag_functions module imported successfully")
    
    print("Testing core module import...")
    from rag_functions.core import config
    print("✓ rag_functions.core.config imported successfully")
    
    print("Testing config functions...")
    from rag_functions.core.config import get_config
    config_obj = get_config()
    print("✓ get_config() function works")
    
    print("All basic imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}") 