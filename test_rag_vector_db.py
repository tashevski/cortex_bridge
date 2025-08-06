#!/usr/bin/env python3
"""Test script for RAG vector database functionality"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_functions.utils.retrieval import (
    search_cue_cards, 
    search_adaptive_prompts, 
    get_all_cue_cards, 
    get_all_adaptive_prompts,
    get_rag_stats
)

def test_rag_vector_db():
    """Test the RAG vector database functionality"""
    print("🧪 Testing RAG Vector Database Functionality")
    print("=" * 50)
    
    # Test 1: Get statistics
    print("\n1. 📊 Getting RAG Statistics:")
    stats = get_rag_stats()
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    else:
        print(f"✓ Total cue cards: {stats['total_cue_cards']}")
        print(f"✓ Total adaptive prompts: {stats['total_adaptive_prompts']}")
        print(f"✓ Total RAG items: {stats['total_rag_items']}")
        
        if stats['prompt_types']:
            print("\n   Prompt types:")
            for prompt_type, count in stats['prompt_types'].items():
                print(f"   - {prompt_type}: {count}")
        
        if stats['medical_issues']:
            print("\n   Medical issues:")
            for issue, count in stats['medical_issues'].items():
                print(f"   - {issue}: {count}")
    
    # Test 2: Search for cue cards
    print("\n2. 🔍 Searching for cue cards:")
    cue_cards = search_cue_cards(query="medical", top_k=5)
    if cue_cards:
        print(f"✓ Found {len(cue_cards)} cue cards")
        for i, card in enumerate(cue_cards[:3]):  # Show first 3
            print(f"   {i+1}. {card['metadata'].get('prompt_type', 'Unknown type')}")
            print(f"      Q: {card['metadata'].get('question', 'N/A')[:100]}...")
            print(f"      A: {card['metadata'].get('answer', 'N/A')[:100]}...")
    else:
        print("   No cue cards found")
    
    # Test 3: Search for adaptive prompts
    print("\n3. 🔍 Searching for adaptive prompts:")
    adaptive_prompts = search_adaptive_prompts(query="", top_k=5)
    if adaptive_prompts:
        print(f"✓ Found {len(adaptive_prompts)} adaptive prompts")
        for i, prompt in enumerate(adaptive_prompts[:3]):  # Show first 3
            print(f"   {i+1}. Issue: {prompt['metadata'].get('medical_issue', 'Unknown')}")
            print(f"      Prompt: {prompt['content'][:100]}...")
    else:
        print("   No adaptive prompts found")
    
    # Test 4: Get all cue cards by prompt type
    print("\n4. 📋 Getting all cue cards by prompt type:")
    family_cards = get_all_cue_cards(prompt_type="medical and care advice for family")
    print(f"✓ Family advice cue cards: {len(family_cards)}")
    
    # Test 5: Get all adaptive prompts
    print("\n5. 📋 Getting all adaptive prompts:")
    all_prompts = get_all_adaptive_prompts()
    print(f"✓ Total adaptive prompts: {len(all_prompts)}")
    
    print("\n✅ RAG Vector Database testing complete!")


if __name__ == "__main__":
    test_rag_vector_db() 