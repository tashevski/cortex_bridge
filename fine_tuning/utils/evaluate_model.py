#!/usr/bin/env python3
"""Evaluation script for fine-tuned Gemma models"""

import json, subprocess, time, argparse
from typing import List, Dict, Any

def test_ollama_model(model: str, prompts: List[str]) -> Dict[str, Any]:
    """Test Ollama model with prompts"""
    results = {"model": model, "responses": [], "avg_time": 0, "success_rate": 0}
    total_time = successful = 0
    
    print(f"🧪 Testing {model}")
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:30]}...")
        
        try:
            start = time.time()
            result = subprocess.run(["ollama", "run", model, prompt], 
                                  capture_output=True, text=True, timeout=60)
            duration = time.time() - start
            total_time += duration
            
            if result.returncode == 0:
                response = result.stdout.strip()
                successful += 1
                results["responses"].append({"prompt": prompt, "response": response, 
                                          "time": duration, "success": True})
                print(f"    ✅ ({duration:.1f}s) {response[:50]}...")
            else:
                results["responses"].append({"prompt": prompt, "response": f"Error: {result.stderr}", 
                                          "time": duration, "success": False})
                print(f"    ❌ {result.stderr}")
        except Exception as e:
            results["responses"].append({"prompt": prompt, "response": f"Error: {e}", 
                                      "time": 0, "success": False})
            print(f"    💥 {e}")
    
    results["avg_time"] = total_time / len(prompts) if prompts else 0
    results["success_rate"] = successful / len(prompts) if prompts else 0
    return results

def get_test_prompts() -> List[str]:
    """Default test prompts"""
    return ["Hello, can you help me?", "What is the capital of France?", 
            "Explain AI in simple terms", "Write a short poem about technology",
            "What should I do if overwhelmed?", "Job interview tips?", 
            "ML vs deep learning?", "Plan a birthday party", 
            "Healthy breakfast ideas", "Stay motivated while learning?"]

def compare_models(models: List[str], prompts: List[str]) -> Dict[str, Any]:
    """Compare multiple models"""
    comparison = {"prompts": prompts, "models": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    for model in models:
        print(f"\n{'='*40}")
        results = test_ollama_model(model, prompts)
        comparison["models"][model] = results
        print(f"📊 {model}: {results['success_rate']:.1%} success, {results['avg_time']:.1f}s avg")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma models")
    parser.add_argument("models", nargs="+", help="Model names to evaluate")
    parser.add_argument("--output", "-o", default="evaluation_results.json")
    parser.add_argument("--custom-prompts", help="JSON file with custom prompts")
    args = parser.parse_args()
    
    # Get prompts
    if args.custom_prompts:
        with open(args.custom_prompts) as f:
            prompts = json.load(f).get("prompts", [])
    else:
        prompts = get_test_prompts()
    
    print(f"🎯 Evaluating {len(args.models)} model(s) with {len(prompts)} prompts")
    
    # Run evaluation
    if len(args.models) == 1:
        results = {"single_evaluation": test_ollama_model(args.models[0], prompts), 
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    else:
        results = compare_models(args.models, prompts)
    
    # Save and display results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {args.output}")
    
    if len(args.models) > 1:
        print("\n🏆 Comparison:")
        for name, data in results["models"].items():
            print(f"{name:25} | {data['success_rate']:6.1%} | {data['avg_time']:5.1f}s")

if __name__ == "__main__":
    main()