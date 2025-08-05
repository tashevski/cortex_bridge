#!/usr/bin/env python3
"""Test script for Gemma integration in RAG functions"""

import sys
from pathlib import Path

# Add program_files to path
sys.path.append(str(Path(__file__).parent.parent / "program_files"))

def test_gemma_client():
    """Test basic Gemma client functionality"""
    print("🧪 Testing Gemma Client...")
    
    try:
        from ai.gemma_client import GemmaClient
        client = GemmaClient()
        
        # Test server availability
        if client.is_server_available():
            print("✅ Ollama server is available")
        else:
            print("❌ Ollama server is not running. Please start it with: ollama serve")
            return False
            
        # Test simple generation
        response = client.generate_response("Hello, this is a test.", timeout=10)
        if response:
            print(f"✅ Basic generation works: {response[:50]}...")
        else:
            print("❌ Failed to generate response")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemma client: {e}")
        return False

def test_rag_analysis():
    """Test RAG analysis with Gemma"""
    print("\n🧪 Testing RAG Analysis...")
    
    try:
        from llm_analysis import analyze_with_llm
        from config import get_config
        
        # Test with minimal data
        test_entities = "Test document with sample entities"
        test_calc = "Sample calculation: 2 + 2 = 4"
        test_references = ["Reference document 1", "Reference document 2"]
        
        # Test with fast config
        config = get_config("fast")
        config.verbose = True
        
        response = analyze_with_llm(test_entities, test_calc, test_references, config)
        
        if response and len(response) > 10:
            print(f"✅ RAG analysis works: {response[:100]}...")
            return True
        else:
            print("❌ RAG analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing RAG analysis: {e}")
        return False

def test_configurations():
    """Test different configuration presets"""
    print("\n🧪 Testing Configuration Presets...")
    
    try:
        from config import get_config, PRESETS
        
        for preset_name in PRESETS:
            config = get_config(preset_name)
            print(f"✅ Config '{preset_name}' loaded: model={config.default_model}, timeout={config.request_timeout}s")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing configurations: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("🚀 Gemma RAG Integration Test Suite")
    print("="*50)
    
    tests = [
        ("Gemma Client", test_gemma_client),
        ("Configurations", test_configurations),
        ("RAG Analysis", test_rag_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n### {test_name} ###")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Gemma integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Pull a Gemma model: ollama pull gemma3n:e4b")
        print("3. Check that program_files path is correct")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)