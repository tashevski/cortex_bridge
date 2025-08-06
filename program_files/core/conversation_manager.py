#!/usr/bin/env python3
"""Conversation management and state handling"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from program_files.utils.text_utils import contains_keywords, truncate_history, format_conversation_context
from program_files.database.enhanced_conversation_db import EnhancedConversationDB
from program_files.config.config import ConversationModeConfig

class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, enable_vector_db: bool = True, config: Optional[ConversationModeConfig] = None):
        if config is None:
            from program_files.config.config import cfg
            config = cfg.conversation_mode
            
        self.config = config
        self.in_gemma_mode = False
        self.waiting_for_feedback = False
        self.gemma_conversation_history = []
        self.last_feedback = None
        self.session_id = self._generate_session_id()
        self.vector_db = EnhancedConversationDB() if enable_vector_db else None
        
        # Emotion tracking for triggering
        self.emotion_history = deque(maxlen=config.emotion_window_size)
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID for a conversation"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    def start_new_conversation(self):
        """Start a new conversation with a fresh session ID"""
        self.session_id = self._generate_session_id()
        self.gemma_conversation_history = []
        print(f"🆕 New conversation session: {self.session_id}")
        
    def is_question(self, text: str) -> bool:
        """Check if text is a question using configurable parameters"""
        text_lower = text.strip().lower()
        
        # Convert config words to lowercase for comparison
        question_words = [w.lower() for w in self.config.question_words]
        aux_prefixes = [p.lower() for p in self.config.auxiliary_prefixes]
        
        return (
            "?" in text
            or any(text_lower.startswith(word) for word in question_words)
            or any(text_lower.startswith(prefix) for prefix in aux_prefixes)
        )
    
    def should_enter_gemma_mode(self, text: str, emotion_text: str = None, confidence: float = None) -> bool:
        """Check if we should enter Gemma conversation mode"""
        # Add emotion to history if provided
        if emotion_text and confidence is not None:
            self.add_emotion_to_history(emotion_text, confidence)
        
        # Check for enter keywords
        keyword_match = contains_keywords(text, self.config.enter_keywords)
        
        # Check if questions should trigger entry
        question_trigger = self.config.enter_on_questions and self.is_question(text)
        
        # Check if emotions should trigger entry
        emotion_trigger = self.should_enter_on_emotion()
        
        return keyword_match or question_trigger or emotion_trigger
    
    def should_exit_gemma_mode(self, text: str) -> bool:
        """Check if we should exit Gemma conversation mode using both keywords and context"""
        # First check for explicit exit keywords
        if contains_keywords(text, self.config.exit_keywords):
            return True
        
        # Check for contextual exit: negative response to help phrase
        return self._is_negative_response_to_help_phrase(text)
    
    def _is_negative_response_to_help_phrase(self, text: str) -> bool:
        """Check if user is responding negatively to a help phrase from the LLM"""
        if not self.gemma_conversation_history:
            return False
        
        text_lower = text.strip().lower()
        
        # Check if current response is negative
        if not contains_keywords(text_lower, self.config.negative_responses):
            return False
        
        # Check if the last assistant message contained a help phrase
        for item in reversed(self.gemma_conversation_history):
            if not item.get('is_user', True):  # Assistant message
                assistant_text = item.get('text', '').lower()
                if contains_keywords(assistant_text, self.config.help_phrases):
                    return True
        
        return False
    

    
    def add_emotion_to_history(self, emotion_text: str, confidence: float):
        """Add emotion data to tracking history"""
        if emotion_text and confidence is not None:
            self.emotion_history.append({
                'emotion': emotion_text.lower(),
                'confidence': confidence,
                'timestamp': datetime.now()
            })
    
    def should_enter_on_emotion(self) -> bool:
        """Check if recent emotions should trigger Gemma mode"""
        if not self.config.enter_on_emotions or not self.emotion_history:
            return False
        
        # Count qualifying emotion instances in recent history
        trigger_count = 0
        for emotion_data in self.emotion_history:
            if (emotion_data['emotion'] in [e.lower() for e in self.config.trigger_emotions] and
                emotion_data['confidence'] >= self.config.emotion_confidence_threshold):
                trigger_count += 1
        
        # Check if we have enough instances to trigger
        return trigger_count >= self.config.emotion_trigger_count
    
    def get_emotion_status(self) -> Dict[str, Any]:
        """Get current emotion tracking status for debugging"""
        if not self.emotion_history:
            return {"status": "no_emotions"}
        
        trigger_emotions = [e.lower() for e in self.config.trigger_emotions]
        qualifying_emotions = [
            e for e in self.emotion_history 
            if e['emotion'] in trigger_emotions and e['confidence'] >= self.config.emotion_confidence_threshold
        ]
        
        return {
            "total_emotions": len(self.emotion_history),
            "qualifying_emotions": len(qualifying_emotions),
            "recent_emotions": [f"{e['emotion']}({e['confidence']:.2f})" for e in list(self.emotion_history)[-3:]],
            "trigger_ready": len(qualifying_emotions) >= self.config.emotion_trigger_count,
            "config": {
                "window_size": self.config.emotion_window_size,
                "trigger_count": self.config.emotion_trigger_count,
                "confidence_threshold": self.config.emotion_confidence_threshold,
                "trigger_emotions": self.config.trigger_emotions
            }
        }
    
    def add_to_history(self, text: str, is_user: bool = True, speaker_name: str = None, audio_features: Optional[Dict] = None, emotion_text: str = None, confidence: float = None, latency_metrics: Optional[Dict] = None, model_used: Optional[str] = None):
        """Add message to conversation history and enhanced vector database"""
        # Always track emotions for potential triggering
        if is_user and emotion_text and confidence is not None:
            self.add_emotion_to_history(emotion_text, confidence)
            
        role = "user" if is_user else "assistant"
        self.gemma_conversation_history.append({"role": role, "content": text})
        self.gemma_conversation_history = truncate_history(
            self.gemma_conversation_history, 
            self.config.max_history_items
        )
        
        # Store in enhanced vector database with audio features
        if self.vector_db:
            self.vector_db.add_conversation_with_audio(
                session_id=self.session_id,
                text=text,
                speaker=speaker_name or "Unknown",
                role=role,
                is_gemma_mode=self.in_gemma_mode,
                audio_features=audio_features,
                feedback=self.last_feedback,
                conversation_context=self.get_conversation_context(),
                emotion_text=emotion_text,
                confidence=confidence,
                latency_metrics=latency_metrics,
                model_used=model_used
            )
    
    def get_conversation_context(self) -> str:
        """Get conversation context for Gemma"""
        return format_conversation_context(
            self.gemma_conversation_history, 
            self.config.max_context_messages
        )
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.in_gemma_mode = False
        self.waiting_for_feedback = False
        self.gemma_conversation_history = []
        self.last_feedback = None