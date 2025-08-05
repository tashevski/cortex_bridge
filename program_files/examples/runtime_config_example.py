"""Example usage of the RuntimeConfigManager"""
from config.runtime_config import runtime_config
from config.config import cfg
import time

def on_speaker_config_change(changes):
    """Callback when speaker detector config changes"""
    print(f"Speaker detector config update:")
    if 'changed' in changes:
        print(f"  - Changed (takes effect immediately): {changes.get('changed', {})}")
    if 'restart_required' in changes:
        print(f"  - Restart required: {changes.get('restart_required', {})}")
    if 'rejected' in changes:
        print(f"  - Rejected: {changes.get('rejected', {})}")

def main():
    print("Runtime Configuration Manager Example\n")
    
    # Register a callback for speaker detector changes
    runtime_config.register_callback('speaker_detector', on_speaker_config_change)
    
    # 1. Get current configuration
    print("1. Current Speaker Detector Config:")
    speaker_config = runtime_config.get_config('speaker_detector')
    print(f"   Similarity threshold: {speaker_config['similarity_threshold']}")
    print(f"   Buffer size: {speaker_config['buffer_size']}")
    print(f"   Min speech energy: {speaker_config['min_speech_energy']}\n")
    
    # 2. Update speaker detector config
    print("2. Updating Speaker Detector parameters...")
    result = runtime_config.update_config('speaker_detector',
        similarity_threshold=0.35,
        buffer_size=32000,  # 2 seconds instead of 1
        min_speech_energy=0.03
    )
    print(f"   Result: {result}\n")
    
    # 3. Update latency monitor config
    print("3. Updating Latency Monitor parameters...")
    result = runtime_config.update_config('latency_monitor',
        high_latency_threshold=5.0,
        acceptable_interruption_rate=0.3
    )
    print(f"   Changed: {result['changed']}")
    print(f"   New high latency threshold: {cfg.latency_monitor.high_latency_threshold}\n")
    
    # 4. Update conversation mode keywords
    print("4. Adding new conversation mode keywords...")
    current_keywords = runtime_config.get_config('conversation_mode')['enter_keywords']
    print(f"   Current enter keywords: {current_keywords}")
    
    new_keywords = current_keywords + ['hello', 'ai']
    result = runtime_config.update_config('conversation_mode',
        enter_keywords=new_keywords,
        enter_on_emotions=True,
        emotion_confidence_threshold=0.8
    )
    print(f"   Changed: {list(result['changed'].keys())}\n")
    
    # 5. Update model selector thresholds
    print("5. Updating Smart Model Selector...")
    result = runtime_config.update_config('smart_model_selector',
        switch_threshold=45,  # 45 seconds instead of 30
        context_length_threshold=1000  # 1000 chars instead of 500
    )
    print(f"   Changed: {result['changed']}\n")
    
    # 6. Testing different types of updates
    print("6. Testing mixed update types...")
    result = runtime_config.update_config('speaker_detector',
        similarity_threshold=0.45,  # Runtime changeable
        use_ecapa_model=False,      # Requires restart
        fft_size=2048              # Read-only
    )
    print(f"   Update result:")
    print(f"     - Changed: {result['changed']}")
    print(f"     - Requires restart: {result['restart_required']}")
    print(f"     - Rejected: {result['rejected']}\n")
    
    # 7. Get all configurations
    print("7. Getting all configurations...")
    all_configs = runtime_config.get_all_configs()
    print(f"   Available components: {list(all_configs.keys())}\n")
    
    # 8. Get parameter information
    print("8. Getting parameter info for speaker_detector...")
    param_info = runtime_config.get_parameter_info('speaker_detector')
    print("   Sample parameters:")
    for param, info in list(param_info.items())[:5]:
        print(f"     - {param}:")
        print(f"       Value: {info['value']}")
        print(f"       Runtime changeable: {info['runtime_changeable']}")
        print(f"       Read-only: {info['read_only']}")
        print(f"       Requires restart: {info['requires_restart']}")
    
    # 9. Reset a component to defaults
    print("\n9. Resetting speaker_detector to defaults...")
    runtime_config.reset_component('speaker_detector')
    speaker_config = runtime_config.get_config('speaker_detector')
    print(f"   Similarity threshold after reset: {speaker_config['similarity_threshold']}")
    
    # Unregister callback
    runtime_config.unregister_callback('speaker_detector', on_speaker_config_change)

if __name__ == "__main__":
    main()