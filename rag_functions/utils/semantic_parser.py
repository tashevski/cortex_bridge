import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "program_files"))

from ai.gemma_client import GemmaClient

def parse_document(text):
    client = GemmaClient()
    # Limit text size to prevent timeout
    limited_text = text[:8000] if len(text) > 8000 else text
    prompt = f"Extract key entities, topics, and sections from the following document. Provide a structured summary:\n\n{limited_text}"
    try:
        return client.generate_response("Parse document", prompt, timeout=90)
    except:
        # Fallback for timeout/connection issues
        return f"Document parsing summary: {len(limited_text)} characters analyzed. Key content: {limited_text[:200]}..."