# retrieval.py - Simple retrieval without langchain dependencies
import re
from typing import List, Optional, Dict, Any

def setup_vector_db(reference_texts, reference_meta=None):
    """Setup simple reference store (fallback without vector embeddings)"""
    references = []
    for i, text in enumerate(reference_texts):
        meta = reference_meta[i] if reference_meta and i < len(reference_meta) else {}
        references.append({
            'text': text,
            'meta': meta,
            'index': i
        })
    return references

def retrieve_references(reference_store, parsed, k=5):
    """Retrieve similar reference documents using simple text matching"""
    if not reference_store:
        return []
    
    # Extract key terms from parsed content
    parsed_text = str(parsed).lower()
    words = re.findall(r'\b\w{4,}\b', parsed_text)  # Words 4+ chars
    word_set = set(words)
    
    # Score references based on word overlap
    scored_refs = []
    for ref in reference_store:
        ref_text = ref['text'].lower()
        ref_words = set(re.findall(r'\b\w{4,}\b', ref_text))
        
        # Calculate simple overlap score
        overlap = len(word_set & ref_words)
        total_words = len(word_set | ref_words)
        score = overlap / total_words if total_words > 0 else 0
        
        scored_refs.append((score, ref['text']))
    
    # Sort by score and return top k
    scored_refs.sort(reverse=True, key=lambda x: x[0])
    return [ref[1] for ref in scored_refs[:k]]