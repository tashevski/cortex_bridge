#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Gemma models
"""

import json
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

def test_ollama_model(model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
    """Test an Ollama model with a set of prompts."""
    results = {
        "model": model_name,
        "responses": [],
        "avg_response_time": 0,
        "success_rate": 0
    }
    
    total_time = 0
    successful_responses = 0
    
    print(f"🧪 Testing model: {model_name}")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"   [{i}/{len(test_prompts)}] Testing prompt: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            
            # Run ollama command
            cmd = ["ollama", "run", model_name, prompt]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60  # 60 second timeout
            )
            
            response_time = time.time() - start_time
            total_time += response_time
            
            if result.returncode == 0:
                response = result.stdout.strip()
                successful_responses += 1
                
                results["responses"].append({
                    "prompt": prompt,
                    "response": response,
                    "response_time": response_time,
                    "success": True
                })
                
                print(f"      ✅ Response ({response_time:.1f}s): {response[:100]}...")
            else:
                results["responses"].append({
                    "prompt": prompt,
                    "response": f"Error: {result.stderr}",
                    "response_time": response_time,
                    "success": False
                })
                print(f"      ❌ Failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            results["responses"].append({
                "prompt": prompt,
                "response": "Error: Timeout",
                "response_time": 60,
                "success": False
            })
            print(f"      ⏰ Timeout")
            
        except Exception as e:
            results["responses"].append({
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "response_time": 0,
                "success": False
            })
            print(f"      💥 Exception: {e}")
    
    results["avg_response_time"] = total_time / len(test_prompts) if test_prompts else 0
    results["success_rate"] = successful_responses / len(test_prompts) if test_prompts else 0
    
    return results

def get_test_prompts() -> List[str]:
    """Get a set of test prompts for evaluation."""
    return [
        "Hello, can you help me?",
        "What is the capital of France?",
        "Explain what artificial intelligence is in simple terms.",
        "Can you write a short poem about technology?",
        "What should I do if I'm feeling overwhelmed?",
        "How do I make a good first impression in a job interview?",
        "What's the difference between machine learning and deep learning?",
        "Can you help me plan a birthday party?",
        "What are some healthy breakfast ideas?",
        "How do I stay motivated while learning something new?"
    ]

def compare_models(models: List[str], test_prompts: List[str]) -> Dict[str, Any]:
    """Compare multiple models on the same test prompts."""
    comparison = {
        "test_prompts": test_prompts,
        "models": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for model in models:
        print(f"\n{'='*50}")
        results = test_ollama_model(model, test_prompts)
        comparison["models"][model] = results
        
        # Print summary
        print(f"\n📊 Summary for {model}:")
        print(f"   Success rate: {results['success_rate']:.1%}")
        print(f"   Avg response time: {results['avg_response_time']:.1f}s")
    
    return comparison

def save_evaluation_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma models")
    parser.add_argument("models", nargs="+", help="Model names to evaluate")
    parser.add_argument("--output", "-o", default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--custom-prompts", help="JSON file with custom test prompts")
    
    args = parser.parse_args()
    
    # Get test prompts
    if args.custom_prompts:
        with open(args.custom_prompts, 'r') as f:
            prompts_data = json.load(f)
            test_prompts = prompts_data.get("prompts", [])
    else:
        test_prompts = get_test_prompts()
    
    print(f"🎯 Evaluating {len(args.models)} model(s) with {len(test_prompts)} prompts")
    
    # Check if models exist
    for model in args.models:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model not in result.stdout:
                print(f"⚠️  Warning: Model '{model}' not found in Ollama")
        except Exception:
            print("⚠️  Warning: Could not check Ollama models")
    
    # Run evaluation
    if len(args.models) == 1:
        results = test_ollama_model(args.models[0], test_prompts)
        evaluation_data = {
            "single_model_evaluation": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        evaluation_data = compare_models(args.models, test_prompts)
    
    # Save results
    save_evaluation_results(evaluation_data, args.output)
    
    # Print final comparison if multiple models
    if len(args.models) > 1:
        print(f"\n🏆 Model Comparison Summary:")
        print("-" * 60)
        
        for model_name, model_results in evaluation_data["models"].items():
            print(f"{model_name:30} | Success: {model_results['success_rate']:6.1%} | "
                  f"Avg Time: {model_results['avg_response_time']:6.1f}s")

if __name__ == "__main__":
    main()