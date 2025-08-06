#!/usr/bin/env python3
"""Test full pipeline integration with database class"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from program_files.core.pipeline_helpers import get_vector_context
from program_files.database.enhanced_conversation_db import EnhancedConversationDB

def test_pipeline_integration():
    """Test pipeline integration with database class"""
    print("🧪 Testing Pipeline Integration with Database Class")
    print("=" * 55)
    
    try:
        # Initialize database
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
        
        print("\n✅ Pipeline integration test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_integration() 