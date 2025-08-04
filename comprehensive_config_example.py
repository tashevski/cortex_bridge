#!/usr/bin/env python3
"""Comprehensive example showing all configurable parameters"""

import sys
sys.path.append('program_files')

from utils.config import (
    Config, 
    SmartModelSelectorConfig, 
    LatencyMonitorConfig, 
    ModelPreloaderConfig,
    GemmaClientConfig,
    cfg
)

def show_all_config_parameters():
    """Display all available configuration parameters"""
    
    print("🔧 Comprehensive Configuration Parameters")
    print("=" * 60)
    
    print("\n📊 SmartModelSelectorConfig:")
    print(f"  switch_threshold: {cfg.smart_model_selector.switch_threshold}s")
    print(f"  context_length_threshold: {cfg.smart_model_selector.context_length_threshold} chars")
    print(f"  complex_keywords: {cfg.smart_model_selector.complex_keywords}")
    print(f"  simple_keywords: {cfg.smart_model_selector.simple_keywords}")
    
    print("\n⏱️  LatencyMonitorConfig:")
    print(f"  history_size: {cfg.latency_monitor.history_size}")
    print(f"  high_latency_threshold: {cfg.latency_monitor.high_latency_threshold}s")
    print(f"  acceptable_interruption_rate: {cfg.latency_monitor.acceptable_interruption_rate:.1%}")
    print(f"  emergency_switch_threshold: {cfg.latency_monitor.emergency_switch_threshold:.1%}")
    print(f"  recent_count_for_interruption_rate: {cfg.latency_monitor.recent_count_for_interruption_rate}")
    print(f"  recent_count_for_avg_response_time: {cfg.latency_monitor.recent_count_for_avg_response_time}")
    
    print("\n🚀 ModelPreloaderConfig:")
    print(f"  base_url: {cfg.model_preloader.base_url}")
    print(f"  timeout: {cfg.model_preloader.timeout}s")
    print(f"  max_retries: {cfg.model_preloader.max_retries}")
    print(f"  background_rotation_interval: {cfg.model_preloader.background_rotation_interval}s")
    print(f"  pull_timeout: {cfg.model_preloader.pull_timeout}s")
    
    print("\n🤖 GemmaClientConfig:")
    print(f"  default_model: {cfg.gemma_client.default_model}")
    print(f"  base_url: {cfg.gemma_client.base_url}")
    print(f"  timeout: {cfg.gemma_client.timeout}s")
    print(f"  stream: {cfg.gemma_client.stream}")

def example_custom_configurations():
    """Show examples of customizing different configurations"""
    
    print("\n🎛️  Custom Configuration Examples")
    print("=" * 60)
    
    # Speed-optimized configuration
    print("\n⚡ Speed-Optimized Configuration:")
    speed_config = Config(
        smart_model_selector=SmartModelSelectorConfig(
            switch_threshold=15,  # Switch faster
            context_length_threshold=200,  # Lower threshold for e4b
            complex_keywords=['analyze', 'research'],  # Fewer complex triggers
            simple_keywords=['what', 'when', 'where', 'yes', 'no', 'quick', 'simple', 'hi', 'hello']
        ),
        latency_monitor=LatencyMonitorConfig(
            high_latency_threshold=2.0,  # More aggressive
            acceptable_interruption_rate=0.1,  # 10% tolerance
            emergency_switch_threshold=0.3,  # 30% emergency threshold
            recent_count_for_interruption_rate=5  # Check fewer recent responses
        ),
        model_preloader=ModelPreloaderConfig(
            timeout=15,  # Faster warming timeout
            background_rotation_interval=180  # More frequent rotation
        ),
        gemma_client=GemmaClientConfig(
            default_model="gemma3n:e2b",  # Start with fast model
            timeout=20  # Shorter timeout
        )
    )
    
    print(f"  Switch threshold: {speed_config.smart_model_selector.switch_threshold}s")
    print(f"  High latency threshold: {speed_config.latency_monitor.high_latency_threshold}s")
    print(f"  Default model: {speed_config.gemma_client.default_model}")
    
    # Quality-optimized configuration
    print("\n🎯 Quality-Optimized Configuration:")
    quality_config = Config(
        smart_model_selector=SmartModelSelectorConfig(
            switch_threshold=60,  # Less frequent switching
            context_length_threshold=800,  # Higher threshold for e4b
            complex_keywords=[
                'analyze', 'explain', 'reasoning', 'complex', 'detailed', 'comprehensive',
                'research', 'investigate', 'study', 'synthesize', 'evaluate'
            ]
        ),
        latency_monitor=LatencyMonitorConfig(
            high_latency_threshold=5.0,  # More tolerant of slow responses
            acceptable_interruption_rate=0.4,  # 40% tolerance
            emergency_switch_threshold=0.7,  # 70% emergency threshold
            history_size=100  # More history for better decisions
        ),
        gemma_client=GemmaClientConfig(
            default_model="gemma3n:e4b",  # Start with capable model
            timeout=60  # Longer timeout for complex queries
        )
    )
    
    print(f"  Switch threshold: {quality_config.smart_model_selector.switch_threshold}s")
    print(f"  Context threshold: {quality_config.smart_model_selector.context_length_threshold} chars")
    print(f"  Default model: {quality_config.gemma_client.default_model}")
    
    # Research-optimized configuration  
    print("\n🔬 Research-Optimized Configuration:")
    research_config = Config(
        smart_model_selector=SmartModelSelectorConfig(
            context_length_threshold=300,
            complex_keywords=[
                'research', 'analyze', 'investigate', 'study', 'examine',
                'hypothesis', 'methodology', 'findings', 'conclusion',
                'literature', 'evidence', 'data', 'statistics'
            ],
            simple_keywords=['what', 'when', 'where', 'define']
        ),
        latency_monitor=LatencyMonitorConfig(
            high_latency_threshold=4.0,
            acceptable_interruption_rate=0.25
        )
    )
    
    print(f"  Research keywords: {research_config.smart_model_selector.complex_keywords[:5]}...")

def show_runtime_modification():
    """Show how to modify configuration at runtime"""
    
    print("\n🔄 Runtime Configuration Modification")
    print("=" * 60)
    
    print("\nOriginal context threshold:", cfg.smart_model_selector.context_length_threshold)
    
    # Modify based on user behavior
    user_is_impatient = True
    if user_is_impatient:
        cfg.smart_model_selector.switch_threshold = 10  # Switch faster
        cfg.latency_monitor.acceptable_interruption_rate = 0.1  # Less tolerance
        print("Adjusted for impatient user:")
        print(f"  Switch threshold: {cfg.smart_model_selector.switch_threshold}s")
        print(f"  Interruption tolerance: {cfg.latency_monitor.acceptable_interruption_rate:.1%}")
    
    # Domain-specific adjustments
    domain = "technical"
    if domain == "technical":
        cfg.smart_model_selector.complex_keywords.extend([
            'algorithm', 'implementation', 'architecture', 'optimization'
        ])
        print(f"Added technical keywords: {cfg.smart_model_selector.complex_keywords[-4:]}")

if __name__ == "__main__":
    show_all_config_parameters()
    example_custom_configurations()
    show_runtime_modification()