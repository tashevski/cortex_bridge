# modules/semantic_parser.py

import sys
from pathlib import Path

# Add program_files to path to import Gemma client
sys.path.append(str(Path(__file__).parent.parent.parent / "program_files"))

from ai.gemma_client import GemmaClient

def parse_document(text):
    """Parse document to extract key entities and structure"""
    client = GemmaClient()
    
    prompt = f"Extract key entities, topics, and sections from the following document. Provide a structured summary:\n\n{text}"
    
    response = client.generate_response(
        prompt=prompt,
        context="Document parsing for entity extraction"
    )
    
    if response is None:
        return "Unable to parse document. Please check if Ollama server is running."
    
    return response
