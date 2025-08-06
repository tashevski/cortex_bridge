#!/usr/bin/env python3
"""Test vector context integration with Gemma responses"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from program_files.core.pipeline_helpers import get_vector_context
from program_files.ai.gemma_client import GemmaClient

def test_vector_context():
    """Test vector context retrieval and integration"""
    print("🧪 Testing Vector Context Integration")
    print("=" * 40)
    
    # Test 1: Get vector context
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
    
    # Test 2: Test with Gemma client (if server available)
    print("\n2. 🤖 Testing with Gemma client:")
    client = GemmaClient()
    
    if client.is_server_available():
        print("✓ Ollama server is available")
        
        # Test with vector context
        test_prompt = "How should I manage diabetes?"
        test_context = "Patient asking about diabetes management"
        
        # Get vector context
        vector_context = get_vector_context(test_prompt)
        
        if vector_context:
            print("   Using vector context in response generation...")
            response = client.generate_response(
                test_prompt, 
                test_context, 
                vector_context=vector_context
            )
            if response:
                print(f"✓ Response generated with vector context")
                print(f"   Response preview: {response[:100]}...")
            else:
                print("   No response generated")
        else:
            print("   No vector context available for test")
    else:
        print("   Ollama server not available (skipping response test)")
    
    print("\n✅ Vector context integration test complete!")

if __name__ == "__main__":
    test_vector_context() 