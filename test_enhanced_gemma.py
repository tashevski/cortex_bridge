#!/usr/bin/env python3
"""Test script for enhanced GemmaClient functionality"""

import sys
import os
sys.path.append('program_files')

from ai.gemma_client import GemmaClient

def test_enhanced_gemma():
    """Test the enhanced GemmaClient features"""
    
    client = GemmaClient("gemma3n:e4b")
    
    # Test 1: Basic functionality (should work as before)
    print("🧪 Test 1: Basic functionality")
    response = client.generate_response("What is the capital of France?")
    print(f"Response: {response}")
    print()
    
    # Test 2: Custom prompt template
    print("🧪 Test 2: Custom prompt template")
    template = "System: You are a helpful assistant.\n{context}\nUser: {prompt}\nAssistant:"
    response = client.generate_response(
        "Explain photosynthesis", 
        context="You are speaking to a 10-year-old child.",
        prompt_template=template
    )
    print(f"Response: {response}")
    print()
    
    # Test 3: Vector context
    print("🧪 Test 3: Vector context")
    vector_data = {
        "previous_messages": [
            {"role": "user", "content": "I'm learning about space"},
            {"role": "assistant", "content": "That's exciting! Space is fascinating."}
        ],
        "user_preferences": {"style": "educational", "level": "beginner"}
    }
    response = client.generate_response(
        "Tell me about black holes",
        vector_context=vector_data
    )
    print(f"Response: {response}")
    print()
    
    # Test 4: All parameters combined
    print("🧪 Test 4: All parameters combined")
    complex_template = """Previous conversation context: {context}

User question: {prompt}

Please provide a helpful response:"""
    
    response = client.generate_response(
        "How do computers work?",
        context="User is interested in technology and prefers simple explanations",
        prompt_template=complex_template,
        vector_context={"topic": "technology", "expertise_level": "beginner"}
    )
    print(f"Response: {response}")
    print()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_enhanced_gemma()