#!/usr/bin/env python3
"""Test config integration with SmartModelSelector"""

import sys
sys.path.append('program_files')

from utils.config import cfg, SmartModelSelectorConfig
from ai.smart_model_selector import SmartModelSelector

def test_config_integration():
    """Test that config integration works correctly"""
    print("🧪 Testing Config Integration")
    print("=" * 40)
    
    # Test 1: Default config
    print("\n📝 Test 1: Default config values")
    selector = SmartModelSelector()
    
    print(f"Switch threshold: {selector.switch_threshold}s")
    print(f"Context threshold: {selector.context_length_threshold} chars")
    print(f"Complex keywords: {selector.complexity_keywords['complex']}")
    print(f"Simple keywords: {selector.complexity_keywords['simple']}")
    
    # Test 2: Modified global config
    print("\n📝 Test 2: Modified global config")
    original_threshold = cfg.smart_model_selector.context_length_threshold
    cfg.smart_model_selector.context_length_threshold = 300
    cfg.smart_model_selector.complex_keywords.append('investigate')
    
    selector2 = SmartModelSelector()
    print(f"Modified context threshold: {selector2.context_length_threshold}")
    print(f"Modified complex keywords: {selector2.complexity_keywords['complex']}")
    
    # Restore original
    cfg.smart_model_selector.context_length_threshold = original_threshold
    cfg.smart_model_selector.complex_keywords.remove('investigate')
    
    # Test 3: Custom config object
    print("\n📝 Test 3: Custom config object")
    custom_config = SmartModelSelectorConfig(
        switch_threshold=15,
        context_length_threshold=100,
        complex_keywords=['research', 'analyze'],
        simple_keywords=['hi', 'bye']
    )
    
    selector3 = SmartModelSelector(custom_config)
    print(f"Custom switch threshold: {selector3.switch_threshold}s")
    print(f"Custom context threshold: {selector3.context_length_threshold}")
    print(f"Custom complex keywords: {selector3.complexity_keywords['complex']}")
    print(f"Custom simple keywords: {selector3.complexity_keywords['simple']}")
    
    # Test 4: Model selection behavior
    print("\n📝 Test 4: Model selection with custom config")
    
    test_cases = [
        ("What time is it?", "", False, "Simple query"),
        ("Please research this topic in detail", "", False, "Research keyword"),
        ("Quick question", "A" * 150, False, "Long context"),
        ("Describe this image", "", True, "Has image"),
    ]
    
    for prompt, context, has_image, description in test_cases:
        should_use_e4b = selector3.should_use_e4b(prompt, context, has_image)
        model = "gemma3n:e4b" if should_use_e4b else "gemma3n:e2b"
        print(f"  {description}: '{prompt}' → {model}")
    
    print("\n✅ Config integration tests completed!")

if __name__ == "__main__":
    test_config_integration()