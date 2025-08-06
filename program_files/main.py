#!/usr/bin/env python3
"""Main entry point for the speech processing pipeline"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path to enable proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apple Silicon specific settings
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for unsupported MPS operations

from program_files.core.program_pipeline import main

if __name__ == "__main__":
    main() 