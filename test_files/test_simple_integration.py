#!/usr/bin/env python3
"""Simple test for pipeline integration"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def get_vector_context(query: str, conversation_context: str = "", top_k: int = 3, vector_db=None):
    """Get relevant vector context from database"""
    try:
        # Use provided vector_db or get from conversation manager
        if vector_db is None:
            return None
        
        # Get vector context using the database class
        context = vector_db.get_vector_context(query, top_k=top_k)
        
        if not context:
            return None
        
        # Format for Gemma client
        return {
            "relevant_cue_cards": [{"q": c.get("question", ""), "a": c.get("answer", "")} for c in context.get("relevant_cue_cards", [])],
            "relevant_prompts": [{"issue": p.get("issue", ""), "prompt": p.get("prompt", "")} for p in context.get("relevant_prompts", [])],
            "similar_conversations": [{"text": c.get("text", ""), "speaker": c.get("speaker", "")} for c in context.get("similar_conversations", [])]
        }
    except Exception as e:
        print(f"Error getting vector context: {e}")
        return None

def test_simple_integration():
    """Test simple integration with database class"""
    print("🧪 Testing Simple Integration with Database Class")
    print("=" * 50)
    
    try:
        # Initialize database
        from program_files.database.enhanced_conversation_db import EnhancedConversationDB
        db = EnhancedConversationDB()
        print("✓ Database initialized successfully")
        
        # Test 1: Test get_vector_context with database
        print("\n1. 🔍 Testing get_vector_context with database:")
        query = "diabetes management"
        context = get_vector_context(query, vector_db=db)
        
        if context:
            print("✓ Vector context retrieved successfully")
            print(f"   - Cue cards: {len(context.get('relevant_cue_cards', []))}")
            print(f"   - Adaptive prompts: {len(context.get('relevant_prompts', []))}")
            print(f"   - Similar conversations: {len(context.get('similar_conversations', []))}")
            
            if context.get('relevant_cue_cards'):
                sample_card = context['relevant_cue_cards'][0]
                print(f"   - Sample cue card: Q: {sample_card['q'][:50]}...")
        else:
            print("   No vector context found")
        
        # Test 2: Test without database (should return None)
        print("\n2. 🔍 Testing get_vector_context without database:")
        context_no_db = get_vector_context(query, vector_db=None)
        if context_no_db is None:
            print("✓ Correctly returns None when no database provided")
        else:
            print("❌ Should return None when no database provided")
        
        # Test 3: Test different queries
        print("\n3. 🔍 Testing different queries:")
        test_queries = ["hypertension", "blood pressure", "medication"]
        
        for query in test_queries:
            context = get_vector_context(query, vector_db=db)
            if context:
                total_items = (len(context.get('relevant_cue_cards', [])) + 
                             len(context.get('relevant_prompts', [])) + 
                             len(context.get('similar_conversations', [])))
                print(f"   ✓ '{query}': {total_items} total items found")
            else:
                print(f"   - '{query}': No results")
        
        print("\n✅ Simple integration test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_integration() 