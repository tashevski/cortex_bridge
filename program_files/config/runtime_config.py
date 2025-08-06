"""Runtime configuration management for dynamic parameter updates"""
from typing import Any, Dict, Optional, List
from program_files.config.config import cfg
import threading, json, os
from dataclasses import fields, is_dataclass, asdict
from pathlib import Path

class RuntimeConfigManager:
    """Manages runtime configuration changes with validation and thread safety"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.change_callbacks = {}
        self.config_file = Path(__file__).parent / "saved_config.json"
        self._defaults = None  # Store original defaults
        
        # Define parameters that require component restart
        self.restart_required_params = {
            'speaker_detector': {'use_ecapa_model', 'model_save_dir'},
            'vosk_model': {'models_base_dir', 'available_models', 'preferred_models'},
            'model_preloader': {'base_url'},
            'gemma_client': {'base_url'},
            'speech_processor': {'sample_rate'},  # Changing sample rate requires reinit
        }
        
        # Define read-only parameters that should never change
        self.read_only_params = {
            # Currently no truly read-only params - most can be changed with care
        }
        
        # Load saved config on startup
        self._save_defaults()
        self.load_config()
        
    def update_config(self, component: str, **kwargs) -> Dict[str, Any]:
        """Update any config component dynamically
        
        Args:
            component: Name of the config component (e.g., 'speaker_detector', 'latency_monitor')
            **kwargs: Parameters to update
            
        Returns:
            Dict with keys:
                - 'changed': Successfully changed values
                - 'restart_required': Parameters that need component restart
                - 'rejected': Parameters that were rejected (read-only or invalid)
        """
        with self.lock:
            # Get the config component
            if not hasattr(cfg, component):
                raise ValueError(f"Unknown config component: {component}")
                
            config_obj = getattr(cfg, component)
            
            # Track different types of changes
            changed = {}
            restart_required = {}
            rejected = {}
            
            # Map component names to validation methods
            validators = {
                'smart_model_selector': self._validate_smart_model_selector,
                'latency_monitor': self._validate_latency_monitor,
                'model_preloader': self._validate_model_preloader,
                'conversation_mode': self._validate_conversation_mode,
                'gemma_client': self._validate_gemma_client,
                'speech_processor': self._validate_speech_processor,
                'speaker_detector': self._validate_speaker_detector,
                'vosk_model': self._validate_vosk_model
            }
            
            # Get validator or use generic validation
            validator = validators.get(component, self._generic_validate)
            
            # Check read-only and restart-required params
            read_only = self.read_only_params.get(component, set())
            restart_params = self.restart_required_params.get(component, set())
            
            # Validate and apply changes
            for key, value in kwargs.items():
                if hasattr(config_obj, key):
                    # Check if parameter is read-only
                    if key in read_only:
                        rejected[key] = f"Read-only parameter (cannot be changed at runtime)"
                        continue
                    
                    # Validate the value
                    validated_value = validator(key, value)
                    if validated_value is not None:
                        setattr(config_obj, key, validated_value)
                        
                        # Categorize the change
                        if key in restart_params:
                            restart_required[key] = validated_value
                        else:
                            changed[key] = validated_value
                    else:
                        rejected[key] = f"Invalid value: {value}"
                else:
                    rejected[key] = f"Unknown parameter"
                        
            # Trigger callbacks if any changes were made
            if component in self.change_callbacks and (changed or restart_required):
                for callback in self.change_callbacks[component]:
                    callback({
                        'changed': changed,
                        'restart_required': restart_required,
                        'rejected': rejected
                    })
                    
            return {
                'changed': changed,
                'restart_required': restart_required,
                'rejected': rejected
            }
    
    def _validate_smart_model_selector(self, key: str, value: Any) -> Any:
        """Validate SmartModelSelector parameters"""
        if key == 'switch_threshold':
            val = int(value)
            return val if val > 0 else None
        elif key == 'context_length_threshold':
            val = int(value)
            return val if val > 0 else None
        elif key in ['complex_keywords', 'simple_keywords']:
            if isinstance(value, list) and all(isinstance(s, str) for s in value):
                return value
        return value
    
    def _validate_latency_monitor(self, key: str, value: Any) -> Any:
        """Validate LatencyMonitor parameters"""
        if key == 'history_size':
            val = int(value)
            return val if val > 0 else None
        elif key == 'high_latency_threshold':
            val = float(value)
            return val if val > 0 else None
        elif key in ['acceptable_interruption_rate', 'emergency_switch_threshold']:
            val = float(value)
            return val if 0.0 <= val <= 1.0 else None
        elif key in ['recent_count_for_interruption_rate', 'recent_count_for_avg_response_time']:
            val = int(value)
            return val if val > 0 else None
        return value
    
    def _validate_model_preloader(self, key: str, value: Any) -> Any:
        """Validate ModelPreloader parameters"""
        if key == 'base_url':
            return str(value) if value else None
        elif key in ['timeout', 'max_retries', 'background_rotation_interval', 'pull_timeout']:
            val = int(value)
            return val if val > 0 else None
        return value
    
    def _validate_conversation_mode(self, key: str, value: Any) -> Any:
        """Validate ConversationMode parameters"""
        if key in ['enter_keywords', 'exit_keywords', 'question_words', 'auxiliary_prefixes', 'trigger_emotions']:
            if isinstance(value, list) and all(isinstance(s, str) for s in value):
                return value
        elif key in ['enter_on_questions', 'enter_on_emotions']:
            return bool(value)
        elif key == 'emotion_confidence_threshold':
            val = float(value)
            return val if 0.0 <= val <= 1.0 else None
        elif key in ['emotion_window_size', 'emotion_trigger_count', 'max_context_messages', 'max_history_items']:
            val = int(value)
            return val if val > 0 else None
        return value
    
    def _validate_gemma_client(self, key: str, value: Any) -> Any:
        """Validate GemmaClient parameters"""
        if key in ['default_model', 'base_url']:
            return str(value) if value else None
        elif key == 'timeout':
            val = int(value)
            return val if val > 0 else None
        elif key == 'stream':
            return bool(value)
        return value
    
    def _validate_speech_processor(self, key: str, value: Any) -> Any:
        """Validate SpeechProcessor parameters"""
        if key == 'sample_rate':
            val = int(value)
            return val if val > 0 else None
        elif key == 'vad_aggressiveness':
            val = int(value)
            return val if 0 <= val <= 3 else None
        elif key == 'silence_threshold':
            val = int(value)
            return val if val > 0 else None
        elif key == 'energy_threshold':
            val = float(value)
            return val if val > 0 else None
        elif key == 'frame_size':
            val = int(value)
            return val if val > 0 else None
        return value
    
    def _validate_speaker_detector(self, key: str, value: Any) -> Any:
        """Validate SpeakerDetector parameters"""
        if key == 'max_speakers':
            val = int(value)
            return val if val > 0 else None
        elif key == 'buffer_size':
            val = int(value)
            return val if val > 0 else None
        elif key in ['similarity_threshold', 'min_speech_energy', 'embedding_alpha']:
            val = float(value)
            return val if 0.0 <= val <= 1.0 else None
        elif key in ['min_frames_for_new_speaker', 'min_frames_for_change']:
            val = int(value)
            return val if val > 0 else None
        elif key in ['use_ecapa_model', 'normalize_embeddings']:
            return bool(value)
        elif key == 'model_save_dir':
            return str(value) if value else None
        elif key in ['fft_size', 'spectral_bands']:
            val = int(value)
            return val if val > 0 else None
        elif key in ['voice_freq_min', 'voice_freq_max']:
            val = float(value)
            return val if val > 0 else None
        return value
    
    def _validate_vosk_model(self, key: str, value: Any) -> Any:
        """Validate VoskModel parameters"""
        if key == 'preferred_models':
            if isinstance(value, list) and all(isinstance(s, str) for s in value):
                return value
        elif key == 'available_models':
            if isinstance(value, dict):
                return value
        elif key in ['models_base_dir', 'fallback_model_name']:
            return str(value) if value else None
        elif key == 'sample_rate':
            val = int(value)
            return val if val > 0 else None
        return value
    
    def _generic_validate(self, key: str, value: Any) -> Any:
        """Generic validation for unknown components"""
        return value
    
    def get_config(self, component: str) -> Dict[str, Any]:
        """Get current configuration for a component"""
        with self.lock:
            if not hasattr(cfg, component):
                raise ValueError(f"Unknown config component: {component}")
                
            config_obj = getattr(cfg, component)
            
            # Convert dataclass to dict
            if is_dataclass(config_obj):
                return {f.name: getattr(config_obj, f.name) for f in fields(config_obj)}
            else:
                # Fallback for non-dataclass objects
                return {k: v for k, v in vars(config_obj).items() if not k.startswith('_')}
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all current configurations"""
        configs = {}
        for field in fields(cfg):
            configs[field.name] = self.get_config(field.name)
        return configs
    
    def register_callback(self, component: str, callback):
        """Register a callback to be called when config changes
        
        Args:
            component: Config component name
            callback: Function that takes a dict of changed values
        """
        if component not in self.change_callbacks:
            self.change_callbacks[component] = []
        self.change_callbacks[component].append(callback)
    
    def unregister_callback(self, component: str, callback):
        """Unregister a callback"""
        if component in self.change_callbacks:
            self.change_callbacks[component].remove(callback)
            if not self.change_callbacks[component]:
                del self.change_callbacks[component]
    
    def reset_component(self, component: str):
        """Reset a component to its default values"""
        with self.lock:
            if not hasattr(cfg, component):
                raise ValueError(f"Unknown config component: {component}")
            
            # Get the field from Config class
            config_field = next((f for f in fields(cfg.__class__) if f.name == component), None)
            if config_field and hasattr(config_field.default_factory, '__call__'):
                # Create new instance with defaults
                new_instance = config_field.default_factory()
                setattr(cfg, component, new_instance)
                
                # Trigger callbacks
                if component in self.change_callbacks:
                    for callback in self.change_callbacks[component]:
                        callback({'reset': True})
    
    def get_parameter_info(self, component: str) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all parameters for a component
        
        Returns dict with parameter names as keys and info dict as values:
            - 'value': Current value
            - 'runtime_changeable': Can be changed without restart
            - 'read_only': Cannot be changed at all
            - 'description': What the parameter does
        """
        with self.lock:
            if not hasattr(cfg, component):
                raise ValueError(f"Unknown config component: {component}")
            
            config_obj = getattr(cfg, component)
            read_only = self.read_only_params.get(component, set())
            restart_params = self.restart_required_params.get(component, set())
            
            info = {}
            if is_dataclass(config_obj):
                for field in fields(config_obj):
                    param_info = {
                        'value': getattr(config_obj, field.name),
                        'runtime_changeable': field.name not in restart_params and field.name not in read_only,
                        'read_only': field.name in read_only,
                        'requires_restart': field.name in restart_params
                    }
                    info[field.name] = param_info
                    
            return info
    
    def _save_defaults(self):
        """Save original default values"""
        if self._defaults is None:
            self._defaults = {}
            for field in fields(cfg):
                component_obj = getattr(cfg, field.name)
                if is_dataclass(component_obj):
                    self._defaults[field.name] = asdict(component_obj)
    
    def save_config(self) -> bool:
        """Save current configuration to disk"""
        try:
            with self.lock:
                config_data = {}
                for field in fields(cfg):
                    component_obj = getattr(cfg, field.name)
                    if is_dataclass(component_obj):
                        config_data[field.name] = asdict(component_obj)
                
                with open(self.config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def load_config(self) -> bool:
        """Load configuration from disk"""
        if not self.config_file.exists():
            return False
            
        try:
            with self.lock:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                for component_name, component_data in config_data.items():
                    if hasattr(cfg, component_name):
                        component_obj = getattr(cfg, component_name)
                        if is_dataclass(component_obj):
                            # Update component with saved values
                            for key, value in component_data.items():
                                if hasattr(component_obj, key):
                                    setattr(component_obj, key, value)
                return True
        except Exception as e:
            print(f"Failed to load config: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset all configuration to original defaults"""
        try:
            with self.lock:
                if self._defaults is None:
                    print("No defaults saved")
                    return False
                
                for component_name, default_data in self._defaults.items():
                    if hasattr(cfg, component_name):
                        component_obj = getattr(cfg, component_name)
                        if is_dataclass(component_obj):
                            # Reset component to defaults
                            for key, value in default_data.items():
                                if hasattr(component_obj, key):
                                    setattr(component_obj, key, value)
                
                # Remove saved config file
                if self.config_file.exists():
                    os.remove(self.config_file)
                
                return True
        except Exception as e:
            print(f"Failed to reset to defaults: {e}")
            return False

# Global instance
runtime_config = RuntimeConfigManager()