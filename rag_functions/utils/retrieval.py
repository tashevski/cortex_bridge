import re

def setup_vector_db(reference_texts, reference_meta=None):
    references = []
    for i, text in enumerate(reference_texts):
        meta = reference_meta[i] if reference_meta and i < len(reference_meta) else {}
        references.append({'text': text, 'meta': meta, 'index': i})
    return references

def retrieve_references(reference_store, parsed, k=5):
    if not reference_store:
        return []
    
    parsed_words = set(re.findall(r'\b\w{4,}\b', str(parsed).lower()))
    
    scored_refs = []
    for ref in reference_store:
        ref_words = set(re.findall(r'\b\w{4,}\b', ref['text'].lower()))
        overlap = len(parsed_words & ref_words)
        total = len(parsed_words | ref_words)
        score = overlap / total if total > 0 else 0
        scored_refs.append((score, ref['text']))
    
    scored_refs.sort(reverse=True, key=lambda x: x[0])
    return [ref[1] for ref in scored_refs[:k]]