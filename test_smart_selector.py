#!/usr/bin/env python3
"""Test the enhanced SmartModelSelector"""

import sys
sys.path.append('program_files')

from ai.smart_model_selector import SmartModelSelector

def test_model_selection():
    """Test various scenarios for model selection"""
    
    # Default selector
    selector = SmartModelSelector()
    
    print("🧪 Testing SmartModelSelector")
    print("=" * 50)
    
    # Test 1: Simple query
    result = selector.should_use_e4b("What time is it?", "")
    print(f"Simple query 'What time is it?' → {'e4b' if result else 'e2b'}")
    
    # Test 2: Complex query
    result = selector.should_use_e4b("Please analyze the complex relationship between...", "")
    print(f"Complex query → {'e4b' if result else 'e2b'}")
    
    # Test 3: Long context
    long_context = "This is a very long conversation history. " * 20  # ~800 chars
    result = selector.should_use_e4b("Continue", long_context)
    print(f"Long context ({len(long_context)} chars) → {'e4b' if result else 'e2b'}")
    
    # Test 4: Image input
    result = selector.should_use_e4b("What's in this image?", "", has_image=True)
    print(f"Image input → {'e4b' if result else 'e2b'}")
    
    # Test 5: Custom keywords
    custom_selector = SmartModelSelector(
        context_length_threshold=200,
        complex_keywords=['research', 'investigate', 'study'],
        simple_keywords=['hi', 'hello', 'thanks']
    )
    
    result = custom_selector.should_use_e4b("Let's research this topic", "")
    print(f"Custom complex keyword 'research' → {'e4b' if result else 'e2b'}")
    
    result = custom_selector.should_use_e4b("Thanks for that", "")
    print(f"Custom simple keyword 'thanks' → {'e4b' if result else 'e2b'}")
    
    # Test 6: Custom context threshold
    medium_context = "Some context. " * 10  # ~140 chars
    result = custom_selector.should_use_e4b("Continue", medium_context)
    print(f"Medium context ({len(medium_context)} chars) with threshold 200 → {'e4b' if result else 'e2b'}")

if __name__ == "__main__":
    test_model_selection()