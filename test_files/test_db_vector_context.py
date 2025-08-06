#!/usr/bin/env python3
"""Test database class vector context functionality"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from program_files.database.enhanced_conversation_db import EnhancedConversationDB

def test_db_vector_context():
    """Test vector context functionality in database class"""
    print("🧪 Testing Database Class Vector Context")
    print("=" * 45)
    
    try:
        # Initialize database
        db = EnhancedConversationDB()
        print("✓ Database initialized successfully")
        
        # Test 1: Get vector context
        print("\n1. 🔍 Testing get_vector_context():")
        query = "diabetes management"
        context = db.get_vector_context(query)
        
        if context:
            print("✓ Vector context retrieved successfully")
            print(f"   - Cue cards: {len(context.get('relevant_cue_cards', []))}")
            print(f"   - Adaptive prompts: {len(context.get('relevant_prompts', []))}")
            print(f"   - Similar conversations: {len(context.get('similar_conversations', []))}")
            
            if context.get('relevant_cue_cards'):
                sample_card = context['relevant_cue_cards'][0]
                print(f"   - Sample cue card: Q: {sample_card['question'][:50]}...")
        else:
            print("   No vector context found (database may be empty)")
        
        # Test 2: Search cue cards directly
        print("\n2. 🔍 Testing search_cue_cards():")
        cue_cards = db.search_cue_cards("diabetes", top_k=3)
        print(f"   Found {len(cue_cards)} cue cards")
        
        if cue_cards:
            sample = cue_cards[0]
            print(f"   - Sample: {sample['question'][:50]}...")
        
        # Test 3: Search adaptive prompts directly
        print("\n3. 🔍 Testing search_adaptive_prompts():")
        adaptive_prompts = db.search_adaptive_prompts("hypertension", top_k=3)
        print(f"   Found {len(adaptive_prompts)} adaptive prompts")
        
        if adaptive_prompts:
            sample = adaptive_prompts[0]
            print(f"   - Sample: {sample['issue']} - {sample['prompt'][:50]}...")
        
        # Test 4: Search conversations directly
        print("\n4. 🔍 Testing search_conversations():")
        conversations = db.search_conversations("medical", top_k=3)
        print(f"   Found {len(conversations)} similar conversations")
        
        if conversations:
            sample = conversations[0]
            print(f"   - Sample: {sample['speaker']} - {sample['text'][:50]}...")
        
        print("\n✅ Database class vector context test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_db_vector_context() 