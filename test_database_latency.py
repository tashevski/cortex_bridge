#!/usr/bin/env python3
"""Test database integration with latency monitoring"""

import sys
import time
import threading
sys.path.append('program_files')

from ai.optimized_gemma_client import OptimizedGemmaClient
from core.conversation_manager import ConversationManager
from utils.enhanced_conversation_db import EnhancedConversationDB

def simulate_conversation_with_latency():
    """Simulate a conversation with various latency scenarios"""
    print("🧪 Testing Database Latency Integration")
    print("=" * 50)
    
    # Initialize components
    client = OptimizedGemmaClient("gemma3n:e2b")
    conversation_manager = ConversationManager()
    conversation_manager.start_new_conversation()
    conversation_manager.in_gemma_mode = True
    
    # Test scenarios
    scenarios = [
        {
            "user_input": "What is 2+2?",
            "simulate_interruption": False,
            "expected_model": "gemma3n:e2b",
            "description": "Simple query, no interruption"
        },
        {
            "user_input": "Please explain quantum computing in detail",
            "simulate_interruption": True,
            "interruption_delay": 1.0,
            "interruption_duration": 2.0,
            "expected_model": "gemma3n:e4b",
            "description": "Complex query with user interruption"
        },
        {
            "user_input": "Quick question about AI",
            "simulate_interruption": True,
            "interruption_delay": 0.5,
            "interruption_duration": 1.5,
            "expected_model": "gemma3n:e2b",
            "description": "Simple query but user impatient"
        },
        {
            "user_input": "Another complex analysis request",
            "simulate_interruption": True,
            "interruption_delay": 0.3,
            "interruption_duration": 3.0,
            "expected_model": "gemma3n:e2b",  # Should switch to fast due to previous interruptions
            "description": "Complex query but system adapts to user impatience"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['description']}")
        print(f"   User input: '{scenario['user_input']}'")
        
        # Add user message to history
        conversation_manager.add_to_history(
            scenario['user_input'], 
            True, 
            "TestUser", 
            emotion_text="neutral",
            confidence=0.8
        )
        
        # Generate response with optional interruption simulation
        def generate_response():
            context = conversation_manager.get_conversation_context()
            response = client.generate_response_optimized(scenario['user_input'], context)
            
            if response:
                print(f"   🤖 Response: {response[:100]}...")
                
                # Get latency metrics and store in database
                latency_metrics = client.get_last_latency_metrics()
                conversation_manager.add_to_history(
                    response, 
                    False, 
                    "Gemma",
                    latency_metrics=latency_metrics
                )
                
                # Print metrics
                if latency_metrics:
                    print(f"   📊 Response time: {latency_metrics['response_time']:.2f}s")
                    print(f"   🤖 Model used: {latency_metrics['model_used']}")
                    print(f"   🔄 Model switched: {latency_metrics['model_switched']}")
                    if latency_metrics['user_spoke_during_response']:
                        print(f"   🗣️  User interrupted for {latency_metrics['speech_activity_during_response']:.1f}s")
        
        # Start response generation
        response_thread = threading.Thread(target=generate_response)
        response_thread.start()
        
        # Simulate user interruption if specified
        if scenario.get('simulate_interruption'):
            def simulate_speech():
                time.sleep(scenario['interruption_delay'])
                print(f"   👤 User starts speaking...")
                
                start_time = time.time()
                while time.time() - start_time < scenario['interruption_duration']:
                    client.record_speech_activity(True)
                    time.sleep(0.1)
                
                client.record_speech_activity(False)
                print(f"   👤 User stops speaking after {scenario['interruption_duration']:.1f}s")
            
            speech_thread = threading.Thread(target=simulate_speech)
            speech_thread.start()
            speech_thread.join()
        
        response_thread.join()
        
        # Show current latency status
        print("   📈 Current latency status:")
        analysis = client.get_latency_status()
        if analysis.get('total_responses', 0) > 0:
            print(f"      Recent interruption rate: {analysis.get('recent_interruption_rate', 0):.1%}")
            print(f"      Should prioritize speed: {analysis.get('should_prioritize_speed', False)}")
    
    # Final database analytics
    print("\n📊 Final Database Analytics:")
    if conversation_manager.vector_db:
        db_analytics = conversation_manager.vector_db.get_latency_analytics()
        if db_analytics.get("status") == "no_data":
            print("   No data stored in database")
        else:
            print(f"   Total responses: {db_analytics['total_responses']}")
            print(f"   Interruption rate: {db_analytics['interruption_rate']:.1%}")
            print(f"   High latency rate: {db_analytics['high_latency_rate']:.1%}")
            print(f"   Model switch rate: {db_analytics['model_switch_rate']:.1%}")
            print(f"   Model usage: {db_analytics['model_usage']}")
            print(f"   Avg response times: {db_analytics['avg_response_times']}")
            
            # Check for problematic sessions
            problematic = conversation_manager.vector_db.get_problematic_sessions()
            if problematic:
                print(f"\n⚠️  Problematic sessions found: {len(problematic)}")
                for session in problematic[:3]:  # Show top 3
                    print(f"      Session {session['session_id']}: {session['interruption_rate']:.1%} interruption rate")
    
    print("\n✅ Database latency integration test complete!")

if __name__ == "__main__":
    simulate_conversation_with_latency()