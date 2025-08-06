import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from program_files.database.enhanced_conversation_db import EnhancedConversationDB
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

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

def extract_medical_issues_list(response_text):
    """Extract medical issues list from LLM response."""
    try:
        # Find list pattern and extract items
        match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
        if match:
            items = [item.strip().strip('"\'') for item in match.group(1).split(',')]
            return [item for item in items if item]
        
        # Fallback: try JSON parsing
        parsed = json.loads(response_text)
        return parsed if isinstance(parsed, list) else [response_text.strip()]
    except:
        return [response_text.strip()]


def get_rag_vector_db():
    """Get the RAG vector database instance"""
    if not VECTOR_DB_AVAILABLE:
        raise ImportError("Vector database not available. Make sure program_files.database.enhanced_conversation_db is accessible.")
    
    # Use the same database as the main program
    base_dir = Path(__file__).parent.parent.parent / "program_files"
    persist_directory = base_dir / "data" / "vector_db"
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    return EnhancedConversationDB(str(persist_directory))


def search_cue_cards(query: str = "", prompt_type: str = None, document_path: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for cue cards in the vector database.
    
    Args:
        query: Search query (empty string for all)
        prompt_type: Filter by prompt type (e.g., "medical and care advice for family")
        document_path: Filter by document path
        top_k: Number of results to return
        
    Returns:
        List of cue card dictionaries with metadata
    """
    if not VECTOR_DB_AVAILABLE:
        return []
    
    try:
        vector_db = get_rag_vector_db()
        
        # Build filter metadata - ChromaDB requires specific operator syntax
        conditions = [{"content_type": {"$eq": "cue_card"}}]
        if prompt_type:
            conditions.append({"prompt_type": {"$eq": prompt_type}})
        if document_path:
            conditions.append({"document_path": {"$eq": str(document_path)}})
        
        # Use $and operator for multiple conditions
        filter_conditions = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        # Search the database
        results = vector_db.conversations.query(
            query_texts=[query] if query else [""],
            n_results=top_k,
            where=filter_conditions
        )
        
        # Format results
        cue_cards = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                cue_cards.append({
                    "content": doc,
                    "metadata": metadata,
                    "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                })
        
        return cue_cards
    
    except Exception as e:
        print(f"Error searching cue cards: {e}")
        return []


def search_adaptive_prompts(query: str = "", medical_issue: str = None, document_path: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for adaptive prompts in the vector database.
    
    Args:
        query: Search query (empty string for all)
        medical_issue: Filter by medical issue
        document_path: Filter by document path
        top_k: Number of results to return
        
    Returns:
        List of adaptive prompt dictionaries with metadata
    """
    if not VECTOR_DB_AVAILABLE:
        return []
    
    try:
        vector_db = get_rag_vector_db()
        
        # Build filter metadata - ChromaDB requires specific operator syntax
        conditions = [{"content_type": {"$eq": "adaptive_prompt"}}]
        if medical_issue:
            conditions.append({"medical_issue": {"$eq": medical_issue}})
        if document_path:
            conditions.append({"document_path": {"$eq": str(document_path)}})
        
        # Use $and operator for multiple conditions
        filter_conditions = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        # Search the database
        results = vector_db.conversations.query(
            query_texts=[query] if query else [""],
            n_results=top_k,
            where=filter_conditions
        )
        
        # Format results
        adaptive_prompts = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                adaptive_prompts.append({
                    "content": doc,
                    "metadata": metadata,
                    "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                })
        
        return adaptive_prompts
    
    except Exception as e:
        print(f"Error searching adaptive prompts: {e}")
        return []


def get_all_cue_cards(prompt_type: str = None, document_path: str = None) -> List[Dict[str, Any]]:
    """Get all cue cards from the database"""
    return search_cue_cards("", prompt_type, document_path, top_k=1000)


def get_all_adaptive_prompts(medical_issue: str = None, document_path: str = None) -> List[Dict[str, Any]]:
    """Get all adaptive prompts from the database"""
    return search_adaptive_prompts("", medical_issue, document_path, top_k=1000)


def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about stored RAG content"""
    if not VECTOR_DB_AVAILABLE:
        return {"error": "Vector database not available"}
    
    try:
        vector_db = get_rag_vector_db()
        
        # Get all data
        all_data = vector_db.conversations.get()
        
        # Count by content type
        cue_cards = [m for m in all_data['metadatas'] if m.get('content_type') == 'cue_card']
        adaptive_prompts = [m for m in all_data['metadatas'] if m.get('content_type') == 'adaptive_prompt']
        
        # Count by prompt type
        prompt_types = {}
        for metadata in cue_cards:
            prompt_type = metadata.get('prompt_type', 'unknown')
            prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
        
        # Count by medical issue
        medical_issues = {}
        for metadata in adaptive_prompts:
            issue = metadata.get('medical_issue', 'unknown')
            medical_issues[issue] = medical_issues.get(issue, 0) + 1
        
        return {
            "total_cue_cards": len(cue_cards),
            "total_adaptive_prompts": len(adaptive_prompts),
            "prompt_types": prompt_types,
            "medical_issues": medical_issues,
            "total_rag_items": len(cue_cards) + len(adaptive_prompts)
        }
    
    except Exception as e:
        return {"error": f"Error getting stats: {e}"}