#!/usr/bin/env python3
"""Test latency monitoring functionality"""

import sys
import time
import threading
sys.path.append('program_files')

from ai.optimized_gemma_client import OptimizedGemmaClient

def simulate_user_speech(client, speech_patterns):
    """Simulate user speech during model response"""
    print("🎙️  Simulating user speech patterns...")
    
    for delay, duration, description in speech_patterns:
        time.sleep(delay)
        print(f"   👤 User starts speaking ({description})")
        
        # Simulate speech activity
        speech_start = time.time()
        while time.time() - speech_start < duration:
            client.record_speech_activity(True)
            time.sleep(0.1)  # 100ms intervals
        
        client.record_speech_activity(False)
        print(f"   👤 User stops speaking after {duration:.1f}s")

def test_latency_monitoring():
    """Test the latency monitoring system"""
    print("🧪 Testing Latency Monitoring System")
    print("=" * 50)
    
    client = OptimizedGemmaClient("gemma3n:e2b")
    
    # Test 1: Normal response (no interruption)
    print("\n📊 Test 1: Normal response")
    response = client.generate_response_optimized("What is 2+2?")
    print(f"Response: {response}")
    
    # Test 2: Response with user interruption
    print("\n📊 Test 2: Response with user interruption")
    
    # Start response generation in thread
    def generate_slow_response():
        return client.generate_response_optimized(
            "Please explain in detail the complex relationship between quantum mechanics and general relativity"
        )
    
    response_thread = threading.Thread(target=generate_slow_response)
    response_thread.start()
    
    # Simulate user speaking during response
    speech_patterns = [
        (1.0, 2.0, "impatient interruption"),  # Start speaking after 1s, speak for 2s
        (0.5, 1.5, "follow-up question")       # Then speak again
    ]
    
    speech_thread = threading.Thread(target=simulate_user_speech, args=(client, speech_patterns))
    speech_thread.start()
    
    response_thread.join()
    speech_thread.join()
    
    # Test 3: Multiple responses to trigger adaptive behavior
    print("\n📊 Test 3: Multiple responses with interruptions")
    for i in range(5):
        print(f"\nResponse {i+1}:")
        
        # Generate response
        response_thread = threading.Thread(
            target=lambda: client.generate_response_optimized(f"Complex question {i+1} requiring detailed analysis")
        )
        response_thread.start()
        
        # Simulate frequent interruptions
        if i < 3:  # First 3 responses get interrupted
            speech_thread = threading.Thread(
                target=simulate_user_speech, 
                args=(client, [(0.5, 1.0, f"interruption {i+1}")])
            )
            speech_thread.start()
            speech_thread.join()
        
        response_thread.join()
        
        # Show status after each response
        print(f"   Interruption rate: {client.latency_monitor.get_interruption_rate(5):.1%}")
    
    # Final status
    print("\n📊 Final Latency Analysis:")
    client.print_latency_status()
    
    # Test emergency mode
    print("\n📊 Testing emergency speed mode:")
    analysis = client.get_latency_status()
    print(f"Should prioritize speed: {analysis.get('should_prioritize_speed', False)}")
    
    # Test model recommendation override
    normal_recommendation = "gemma3n:e4b"
    final_model, reason = client.latency_monitor.get_model_recommendation(normal_recommendation)
    print(f"Model recommendation: {normal_recommendation} → {final_model}")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    test_latency_monitoring()