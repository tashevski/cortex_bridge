#!/usr/bin/env python3
"""Main entry point for the speech processing pipeline"""

import os
# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.program_pipeline import main

if __name__ == "__main__":
    main() 