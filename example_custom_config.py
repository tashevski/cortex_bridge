#!/usr/bin/env python3
"""Example of how to customize SmartModelSelector configuration"""

import sys
sys.path.append('program_files')

from utils.config import Config, SmartModelSelectorConfig, LatencyMonitorConfig, GemmaClientConfig
from ai.optimized_gemma_client import OptimizedGemmaClient

def example_custom_config():
    """Example of customizing configuration"""
    
    # Option 1: Customize the global config
    print("🔧 Customizing global config...")
    from utils.config import cfg
    
    # Modify SmartModelSelector settings
    cfg.smart_model_selector.context_length_threshold = 300  # Lower threshold
    cfg.smart_model_selector.complex_keywords.extend(['research', 'investigate'])
    
    # Modify LatencyMonitor settings
    cfg.latency_monitor.high_latency_threshold = 2.5  # More aggressive
    cfg.latency_monitor.acceptable_interruption_rate = 0.15  # 15% instead of 20%
    
    # Use the customized config
    client = OptimizedGemmaClient()
    print(f"Context threshold: {client.selector.context_length_threshold}")
    print(f"Complex keywords: {client.selector.complexity_keywords['complex']}")
    print(f"Latency threshold: {client.latency_monitor.high_latency_threshold}")
    
    # Option 2: Create custom config objects
    print("\n🔧 Creating custom config objects...")
    
    custom_selector_config = SmartModelSelectorConfig(
        switch_threshold=20,  # Switch faster
        context_length_threshold=200,  # Even lower
        complex_keywords=['analyze', 'research', 'investigate', 'study'],
        simple_keywords=['hi', 'hello', 'thanks', 'bye']
    )
    
    custom_latency_config = LatencyMonitorConfig(
        history_size=100,  # More history
        high_latency_threshold=2.0,  # Very aggressive
        acceptable_interruption_rate=0.1  # 10%
    )
    
    custom_gemma_config = GemmaClientConfig(
        default_model="gemma3n:e4b",  # Start with capable model
        base_url="http://localhost:11434"
    )
    
    # Create client with custom configs (would need to modify constructor)
    print("Custom selector config:")
    print(f"  Switch threshold: {custom_selector_config.switch_threshold}s")
    print(f"  Context threshold: {custom_selector_config.context_length_threshold} chars")
    print(f"  Complex keywords: {custom_selector_config.complex_keywords}")
    
    # Option 3: Runtime config modification
    print("\n🔧 Runtime config modification...")
    
    # Add domain-specific keywords based on usage
    if "research_mode":
        cfg.smart_model_selector.complex_keywords.extend([
            'hypothesis', 'methodology', 'analysis', 'synthesis'
        ])
        cfg.smart_model_selector.context_length_threshold = 200  # Research needs detail
    
    print("Updated for research mode:")
    print(f"  Complex keywords: {cfg.smart_model_selector.complex_keywords}")
    print(f"  Context threshold: {cfg.smart_model_selector.context_length_threshold}")

if __name__ == "__main__":
    example_custom_config()