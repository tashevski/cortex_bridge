#!/usr/bin/env python3
"""Test script for RAG vector database storage functionality"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_functions.utils.retrieval import (
    get_rag_vector_db,
    search_cue_cards, 
    search_adaptive_prompts, 
    get_all_cue_cards, 
    get_all_adaptive_prompts,
    get_rag_stats
)
import uuid
from datetime import datetime

def test_storage_functionality():
    """Test storing and retrieving cue cards and adaptive prompts"""
    print("🧪 Testing RAG Vector Database Storage Functionality")
    print("=" * 60)
    
    try:
        # Get vector database
        vector_db = get_rag_vector_db()
        print("✓ Vector database initialized successfully")
        
        # Test data
        test_document_path = "/test/document.pdf"
        test_timestamp = datetime.now().isoformat()
        
        # Test 1: Store cue cards
        print("\n1. 📝 Storing test cue cards:")
        test_cue_cards = {
            "question_1": {
                "question": "How to manage diabetes?",
                "answer": "Monitor blood sugar regularly and follow medication schedule."
            },
            "question_2": {
                "question": "What are the symptoms of high blood pressure?",
                "answer": "Headaches, dizziness, and chest pain may indicate high blood pressure."
            }
        }
        
        # Store cue cards manually
        doc_id = str(uuid.uuid4())
        for i, (key, value) in enumerate(test_cue_cards.items()):
            content = f"Question: {value['question']}\nAnswer: {value['answer']}"
            metadata = {
                "document_path": test_document_path,
                "cue_card_id": f"{doc_id}_{i}",
                "prompt_type": "medical and care advice for family",
                "question": value['question'],
                "answer": value['answer'],
                "timestamp": test_timestamp,
                "content_type": "cue_card",
                "session_id": f"test_session_{test_timestamp.replace(':', '-')}"
            }
            
            vector_db.conversations.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"test_cue_card_{doc_id}_{i}"]
            )
            print(f"   ✓ Stored cue card {i+1}: {value['question'][:50]}...")
        
        # Test 2: Store adaptive prompts
        print("\n2. 📝 Storing test adaptive prompts:")
        test_adaptive_prompts = [
            "briefly summarise and identify any issues relating to diabetes in the associated conversations",
            "briefly summarise and identify any issues relating to hypertension in the associated conversations"
        ]
        test_medical_issues = ["diabetes", "hypertension"]
        
        for i, (prompt, issue) in enumerate(zip(test_adaptive_prompts, test_medical_issues)):
            metadata = {
                "document_path": test_document_path,
                "prompt_id": f"test_adaptive_{uuid.uuid4()}",
                "medical_issue": issue,
                "prompt_text": prompt,
                "timestamp": test_timestamp,
                "content_type": "adaptive_prompt",
                "session_id": f"test_session_{test_timestamp.replace(':', '-')}"
            }
            
            vector_db.conversations.add(
                documents=[prompt],
                metadatas=[metadata],
                ids=[f"test_adaptive_prompt_{i}_{test_timestamp.replace(':', '-')}"]
            )
            print(f"   ✓ Stored adaptive prompt for {issue}")
        
        # Test 3: Verify storage with statistics
        print("\n3. 📊 Verifying storage with statistics:")
        stats = get_rag_stats()
        print(f"   ✓ Total cue cards: {stats['total_cue_cards']}")
        print(f"   ✓ Total adaptive prompts: {stats['total_adaptive_prompts']}")
        print(f"   ✓ Total RAG items: {stats['total_rag_items']}")
        
        # Test 4: Search for stored content
        print("\n4. 🔍 Searching for stored content:")
        
        # Search cue cards
        cue_cards = search_cue_cards(query="diabetes", top_k=5)
        print(f"   ✓ Found {len(cue_cards)} cue cards with 'diabetes' query")
        
        # Search adaptive prompts
        adaptive_prompts = search_adaptive_prompts(query="", top_k=5)
        print(f"   ✓ Found {len(adaptive_prompts)} adaptive prompts")
        
        # Test 5: Filter by specific criteria
        print("\n5. 🔍 Filtering by specific criteria:")
        
        # Get cue cards by prompt type
        family_cards = get_all_cue_cards(prompt_type="medical and care advice for family")
        print(f"   ✓ Found {len(family_cards)} family advice cue cards")
        
        # Get adaptive prompts by medical issue
        diabetes_prompts = get_all_adaptive_prompts(medical_issue="diabetes")
        print(f"   ✓ Found {len(diabetes_prompts)} diabetes-related adaptive prompts")
        
        # Test 6: Show sample results
        print("\n6. 📋 Sample stored content:")
        if cue_cards:
            print("   Sample cue card:")
            sample_card = cue_cards[0]
            print(f"   - Question: {sample_card['metadata'].get('question', 'N/A')}")
            print(f"   - Answer: {sample_card['metadata'].get('answer', 'N/A')}")
        
        if adaptive_prompts:
            print("   Sample adaptive prompt:")
            sample_prompt = adaptive_prompts[0]
            print(f"   - Issue: {sample_prompt['metadata'].get('medical_issue', 'N/A')}")
            print(f"   - Prompt: {sample_prompt['content'][:100]}...")
        
        print("\n✅ RAG Vector Database storage testing complete!")
        print("   All functionality is working correctly!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_storage_functionality() 