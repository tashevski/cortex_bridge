#!/usr/bin/env python3
"""Simple test for vector context integration"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def get_vector_context(query: str, conversation_context: str = "", top_k: int = 3):
    """Get relevant vector context from database"""
    try:
        from rag_functions.utils.retrieval import search_cue_cards, search_adaptive_prompts
        
        # Search for relevant content
        cue_cards = search_cue_cards(query, top_k=top_k)
        adaptive_prompts = search_adaptive_prompts(query, top_k=top_k)
        
        if not cue_cards and not adaptive_prompts:
            return None
        
        return {
            "relevant_cue_cards": [{"q": c["metadata"].get("question", ""), "a": c["metadata"].get("answer", "")} for c in cue_cards],
            "relevant_prompts": [{"issue": p["metadata"].get("medical_issue", ""), "prompt": p["content"]} for p in adaptive_prompts]
        }
    except Exception as e:
        print(f"Error getting vector context: {e}")
        return None

def test_vector_context():
    """Test vector context retrieval"""
    print("🧪 Testing Vector Context Integration")
    print("=" * 40)
    
    # Test vector context retrieval
    print("\n1. 🔍 Testing vector context retrieval:")
    query = "diabetes management"
    context = get_vector_context(query)
    
    if context:
        print("✓ Vector context retrieved successfully")
        print(f"   - Cue cards found: {len(context.get('relevant_cue_cards', []))}")
        print(f"   - Adaptive prompts found: {len(context.get('relevant_prompts', []))}")
        
        if context.get('relevant_cue_cards'):
            sample_card = context['relevant_cue_cards'][0]
            print(f"   - Sample cue card: Q: {sample_card['q'][:50]}...")
        
        if context.get('relevant_prompts'):
            sample_prompt = context['relevant_prompts'][0]
            print(f"   - Sample prompt: {sample_prompt['issue']}")
    else:
        print("   No vector context found (database may be empty)")
    
    # Test with different queries
    print("\n2. 🔍 Testing different queries:")
    test_queries = ["hypertension", "blood pressure", "medication"]
    
    for query in test_queries:
        context = get_vector_context(query)
        if context:
            print(f"   ✓ '{query}': {len(context.get('relevant_cue_cards', []))} cards, {len(context.get('relevant_prompts', []))} prompts")
        else:
            print(f"   - '{query}': No results")
    
    print("\n✅ Vector context integration test complete!")

if __name__ == "__main__":
    test_vector_context() 