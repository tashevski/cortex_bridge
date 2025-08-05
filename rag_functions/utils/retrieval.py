# retrieval.py - Simple text-based reference retrieval
import re

def setup_vector_db(reference_texts, reference_meta=None):
    """Setup reference store with metadata"""
    references = []
    for i, text in enumerate(reference_texts):
        meta = reference_meta[i] if reference_meta and i < len(reference_meta) else {}
        references.append({'text': text, 'meta': meta, 'index': i})
    return references

def retrieve_references(reference_store, parsed, k=5):
    """Retrieve most relevant references using keyword matching"""
    if not reference_store:
        return []
    
    # Extract keywords from parsed content
    parsed_words = set(re.findall(r'\b\w{4,}\b', str(parsed).lower()))
    
    # Score references by keyword overlap
    scored_refs = []
    for ref in reference_store:
        ref_words = set(re.findall(r'\b\w{4,}\b', ref['text'].lower()))
        overlap = len(parsed_words & ref_words)
        total = len(parsed_words | ref_words)
        score = overlap / total if total > 0 else 0
        scored_refs.append((score, ref['text']))
    
    # Return top k references
    scored_refs.sort(reverse=True, key=lambda x: x[0])
    return [ref[1] for ref in scored_refs[:k]]