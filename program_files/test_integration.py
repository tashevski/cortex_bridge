#!/usr/bin/env python3
"""Test script for pipeline integration with adaptive monitoring"""

import sys
import time

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from ai.adaptive_system_monitor import adaptive_monitor, SystemMode
        print("✓ AdaptiveSystemMonitor imported")
        
        from core.pipeline_helpers import handle_special_commands
        print("✓ Pipeline helpers imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_mode_transitions():
    """Test mode transitions in the adaptive monitor"""
    print("\nTesting mode transitions...")
    
    from ai.adaptive_system_monitor import adaptive_monitor, SystemMode
    
    # Test mode transitions
    transitions = [
        (SystemMode.IDLE, "System startup"),
        (SystemMode.LISTENING, "Ready for input"),
        (SystemMode.PROCESSING, "Processing speech"),
        (SystemMode.GEMMA, "LLM active"),
        (SystemMode.LISTENING, "LLM complete"),
        (SystemMode.SHUTDOWN, "System shutdown")
    ]
    
    for mode, context in transitions:
        adaptive_monitor.set_system_mode(mode, context)
        current = adaptive_monitor.get_system_mode()
        monitoring_allowed = adaptive_monitor.is_monitoring_allowed()
        
        status_symbol = "✓" if monitoring_allowed else "⏸️"
        print(f"  {status_symbol} Mode: {current.value} - {context}")
        
        if mode == SystemMode.GEMMA and monitoring_allowed:
            print("  ✗ ERROR: Monitoring should be paused in GEMMA mode!")
            return False
    
    print("✓ All mode transitions work correctly")
    return True

def test_monitoring_status():
    """Test monitoring status reporting"""
    print("\nTesting monitoring status...")
    
    from ai.adaptive_system_monitor import adaptive_monitor
    
    # Get status report
    status = adaptive_monitor.get_status_report()
    
    required_fields = ['system_mode', 'monitoring_allowed', 'mode_duration_seconds']
    for field in required_fields:
        if field not in status:
            print(f"✗ Missing field in status: {field}")
            return False
    
    print(f"✓ Status report complete:")
    print(f"  - Mode: {status['system_mode']}")
    print(f"  - Monitoring allowed: {status['monitoring_allowed']}")
    print(f"  - Duration: {status['mode_duration_seconds']:.1f}s")
    
    return True

def test_special_commands():
    """Test special command handling"""
    print("\nTesting special command integration...")
    
    from core.pipeline_helpers import handle_special_commands
    
    # Mock objects (minimal implementation for testing)
    class MockGemmaClient:
        def print_latency_status(self):
            print("  📊 Mock latency status printed")
    
    class MockConversationManager:
        def __init__(self):
            self.vector_db = None
    
    gemma_client = MockGemmaClient()
    conversation_manager = MockConversationManager()
    
    # Test monitoring status command
    result = handle_special_commands("monitoring status", gemma_client, conversation_manager)
    if not result:
        print("✗ 'monitoring status' command not handled")
        return False
    
    print("✓ Special commands work correctly")
    return True

def test_pipeline_integration():
    """Test that pipeline can be imported without running it"""
    print("\nTesting pipeline integration...")
    
    try:
        # Import pipeline components (but don't run main)
        from core.program_pipeline import process_text, load_vosk_model
        print("✓ Pipeline functions imported")
        
        # Test that adaptive monitor is accessible
        from core.program_pipeline import adaptive_monitor
        print("✓ Adaptive monitor accessible from pipeline")
        
        return True
    except ImportError as e:
        print(f"✗ Pipeline integration failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Pipeline Integration Tests")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Mode Transitions", test_mode_transitions),
        ("Monitoring Status", test_monitoring_status),
        ("Special Commands", test_special_commands),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nThe adaptive monitoring system is successfully integrated!")
        print("\nNew voice commands available:")
        print("  - 'monitoring status' - Show current monitoring state")
        print("  - 'latency status' - Show latency metrics")
        print("  - 'database analytics' - Show database analytics")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)