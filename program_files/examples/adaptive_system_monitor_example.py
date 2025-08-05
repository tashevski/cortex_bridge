#!/usr/bin/env python3
"""Example demonstrating the AdaptiveSystemMonitor"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ai.adaptive_system_monitor import AdaptiveSystemMonitor, SystemMode, adaptive_monitor
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def test_system_monitor():
    """Test the adaptive system monitor functionality"""
    
    print("Adaptive System Monitor Testing")
    print("=" * 50)
    
    # Initialize monitor
    monitor = AdaptiveSystemMonitor()
    
    print("\n1. Collecting System Metrics")
    print("-" * 30)
    
    # Collect current metrics
    metrics = monitor.collect_system_metrics(time_window_minutes=60)  # Last hour
    
    print(f"Metrics collected:")
    print(f"  Average response time: {metrics.avg_response_time:.2f}s")
    print(f"  Error rate: {metrics.error_rate:.2%}")
    print(f"  Speaker changes/min: {metrics.speaker_change_frequency:.2f}")
    print(f"  Interruption rate: {metrics.interruption_rate:.2%}")
    print(f"  Successful responses: {metrics.successful_responses}")
    print(f"  Failed responses: {metrics.failed_responses}")
    
    print("\n2. Performance Analysis")
    print("-" * 30)
    
    # Analyze performance
    analysis = monitor.analyze_performance(metrics)
    
    print(f"Performance analysis:")
    print(f"  Response time trend: {analysis.response_time_trend}")
    print(f"  Error trend: {analysis.error_trend}")
    print(f"  Speaker detection accuracy: {analysis.speaker_detection_accuracy:.2%}")
    print(f"  Model efficiency: {analysis.model_efficiency:.2%}")
    print(f"  User satisfaction proxy: {analysis.user_satisfaction_proxy:.2%}")
    print(f"  Bottlenecks: {', '.join(analysis.bottlenecks) if analysis.bottlenecks else 'None'}")
    
    print("\n3. Parameter Optimization")
    print("-" * 30)
    
    # Test parameter optimization
    optimizations = monitor.optimize_parameters(metrics, analysis)
    
    if optimizations:
        print(f"Applied optimizations:")
        for component, changes in optimizations.items():
            if changes.get('changed'):
                print(f"  {component}: {changes['changed']}")
            if changes.get('restart_required'):
                print(f"  {component} (restart required): {changes['restart_required']}")
    else:
        print("No optimizations needed at this time.")
    
    print("\n4. Status Report")
    print("-" * 30)
    
    # Get status report
    status = monitor.get_status_report()
    print(json.dumps(status, indent=2, default=str))

def test_mode_awareness():
    """Test system mode awareness and monitoring behavior"""
    
    print("\n" + "=" * 50)
    print("System Mode Awareness Test")
    print("=" * 50)
    
    monitor = AdaptiveSystemMonitor()
    
    # Test different modes
    print("Testing different system modes and monitoring behavior:\n")
    
    modes_to_test = [
        (SystemMode.IDLE, "System startup"),
        (SystemMode.LISTENING, "Waiting for user input"),
        (SystemMode.PROCESSING, "Processing speech"),
        (SystemMode.GEMMA, "LLM generating response"),
        (SystemMode.LISTENING, "Back to listening"),
        (SystemMode.SHUTDOWN, "System shutdown")
    ]
    
    for mode, context in modes_to_test:
        print(f"Setting mode: {mode.value} ({context})")
        monitor.set_system_mode(mode, context)
        
        # Check monitoring status
        monitoring_allowed = monitor.is_monitoring_allowed()
        current_mode = monitor.get_system_mode()
        
        print(f"  Current mode: {current_mode.value}")
        print(f"  Monitoring allowed: {'✓' if monitoring_allowed else '✗'}")
        
        if not monitoring_allowed:
            print(f"  → Monitoring PAUSED (mode: {mode.value})")
        else:
            print(f"  → Monitoring ACTIVE")
        
        time.sleep(1)  # Small delay to show transitions
        print()
    
    # Show mode history
    status = monitor.get_status_report()
    if status.get('recent_mode_transitions'):
        print("Recent mode transitions:")
        for transition in status['recent_mode_transitions']:
            print(f"  {transition['from']} → {transition['to']} "
                  f"({transition.get('context', 'no context')})")

def test_continuous_monitoring():
    """Test continuous monitoring with mode changes"""
    
    print("\n" + "=" * 50)
    print("Continuous Monitoring with Mode Changes")
    print("=" * 50)
    
    # Use the global monitor instance
    monitor = adaptive_monitor
    
    print("Starting continuous monitoring with simulated mode changes...")
    print("Watch how monitoring pauses during Gemma mode!")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate system operation with mode changes
    simulation_steps = [
        (SystemMode.LISTENING, "Waiting for input", 3),
        (SystemMode.PROCESSING, "Processing speech", 2),
        (SystemMode.GEMMA, "LLM thinking", 4),  # Monitoring should pause here
        (SystemMode.LISTENING, "Back to listening", 3),
        (SystemMode.PROCESSING, "Processing again", 2),
        (SystemMode.GEMMA, "Another LLM response", 3),  # Pause again
        (SystemMode.LISTENING, "Final listening", 2),
    ]
    
    for mode, context, duration in simulation_steps:
        print(f"\n--- Setting mode: {mode.value} ({context}) for {duration}s ---")
        monitor.set_system_mode(mode, context)
        
        for i in range(duration):
            time.sleep(1)
            status = monitor.get_status_report()
            monitoring_status = "ACTIVE" if status.get('monitoring_allowed') else "PAUSED"
            print(f"[{i+1}s] Mode: {status.get('system_mode')}, Monitoring: {monitoring_status}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\nMonitoring stopped.")

def test_gemma_mode_protection():
    """Demonstrate how Gemma mode protects LLM from interference"""
    
    print("\n" + "=" * 50)
    print("Gemma Mode Protection Test")
    print("=" * 50)
    
    monitor = AdaptiveSystemMonitor()
    
    print("Simulating system optimization attempts during different modes:\n")
    
    # Test optimization during normal mode
    print("1. During LISTENING mode:")
    monitor.set_system_mode(SystemMode.LISTENING, "Ready for input")
    if monitor.is_monitoring_allowed():
        print("   ✓ Optimization allowed - system can adjust parameters")
    else:
        print("   ✗ Optimization blocked")
    
    # Test optimization during Gemma mode
    print("\n2. During GEMMA mode (LLM active):")
    monitor.set_system_mode(SystemMode.GEMMA, "LLM processing request")
    if monitor.is_monitoring_allowed():
        print("   ✓ Optimization allowed")
    else:
        print("   ✗ Optimization BLOCKED - protecting LLM performance")
        print("   → Parameter changes postponed until LLM finishes")
    
    # Test returning to normal mode
    print("\n3. After GEMMA mode:")
    monitor.set_system_mode(SystemMode.LISTENING, "LLM finished, back to listening")
    if monitor.is_monitoring_allowed():
        print("   ✓ Optimization resumed - can apply any pending changes")
    else:
        print("   ✗ Optimization blocked")
    
    print("\nThis ensures LLM gets full system resources without interference!")

def simulate_performance_data():
    """Simulate some performance data in the database for testing"""
    print("\n" + "=" * 50)
    print("Performance Data Simulation")
    print("=" * 50)
    print("Note: This would normally be populated by actual system usage.")
    print("For testing, you can manually add data to the database or run the main program.")
    
    # In a real scenario, you might want to add some test data here
    # This is just a placeholder to show how the system would work
    
    from datetime import datetime, timedelta
    import random
    
    print("\nExample of what performance data might look like:")
    
    # Simulate some metrics over time
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(10):
        timestamp = base_time + timedelta(minutes=i*6)
        response_time = random.uniform(0.5, 3.0)
        error_rate = random.uniform(0.0, 0.15)
        
        print(f"  {timestamp.strftime('%H:%M')}: "
              f"Response: {response_time:.2f}s, "
              f"Errors: {error_rate:.1%}")

if __name__ == "__main__":
    print("🤖 Adaptive System Monitor Demo")
    print("This system monitors program performance and adjusts parameters automatically.")
    print("\nNew Features:")
    print("- MODE AWARENESS: Respects system operational modes")
    print("- GEMMA PROTECTION: Pauses optimization during LLM processing")
    print("- Collects metrics from conversation database")
    print("- Analyzes performance trends")
    print("- Automatically optimizes parameters")
    print("- Prevents oscillation with cooldown periods")
    print("- Runs continuously in background")
    
    try:
        # Test basic functionality
        test_system_monitor()
        
        # Test mode awareness
        test_mode_awareness()
        
        # Test Gemma mode protection
        test_gemma_mode_protection()
        
        # Simulate what data might look like
        simulate_performance_data()
        
        # Test continuous monitoring (optional)
        response = input("\nWould you like to test continuous monitoring with mode changes? (y/n): ")
        if response.lower() == 'y':
            test_continuous_monitoring()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logging.exception("Full error details")
    
    print("\n✅ Demo complete!")
    print("\nTo integrate into your main program:")
    print("  from ai.adaptive_system_monitor import adaptive_monitor, SystemMode")
    print("  adaptive_monitor.start_monitoring()  # Start background monitoring")
    print("  ")
    print("  # Set mode when entering LLM processing:")
    print("  adaptive_monitor.set_system_mode(SystemMode.GEMMA, 'Processing user query')")
    print("  # ... LLM processes ...")
    print("  adaptive_monitor.set_system_mode(SystemMode.LISTENING, 'Ready for next input')")
    print("  ")
    print("  adaptive_monitor.stop_monitoring()   # Clean shutdown")