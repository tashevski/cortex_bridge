#!/usr/bin/env python3
"""Main entry point for the speech processing pipeline"""

import os
# Suppress tokenizers parallelism warnings

# Apple Silicon specific settings
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for unsupported MPS operations


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.program_pipeline import main

if __name__ == "__main__":
    main() 