#!/usr/bin/env python3
"""Conversation management and state handling"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.text_utils import is_question, contains_keywords, truncate_history, format_conversation_context
from utils.enhanced_conversation_db import EnhancedConversationDB

ENTER_KEYWORDS = ['hey gemma', 'gemma', 'assistant', 'help']
EXIT_KEYWORDS = ['exit', 'quit', 'stop', 'bye', 'goodbye', 'end conversation']

class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, enable_vector_db: bool = True):
        self.in_gemma_mode = False
        self.waiting_for_feedback = False
        self.gemma_conversation_history = []
        self.last_feedback = None
        self.session_id = self._generate_session_id()
        self.vector_db = EnhancedConversationDB() if enable_vector_db else None
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID for a conversation"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    def start_new_conversation(self):
        """Start a new conversation with a fresh session ID"""
        self.session_id = self._generate_session_id()
        self.gemma_conversation_history = []
        print(f"🆕 New conversation session: {self.session_id}")
        
    def should_enter_gemma_mode(self, text: str) -> bool:
        """Check if we should enter Gemma conversation mode"""
        text_lower = text.lower().strip()
        return (contains_keywords(text_lower, ENTER_KEYWORDS) or 
                is_question(text))
    
    def should_exit_gemma_mode(self, text: str) -> bool:
        """Check if we should exit Gemma conversation mode"""
        return contains_keywords(text.lower().strip(), EXIT_KEYWORDS)
    
    def add_to_history(self, text: str, is_user: bool = True, speaker_name: str = None, audio_features: Optional[Dict] = None):
        """Add message to conversation history and enhanced vector database"""
        role = "user" if is_user else "assistant"
        self.gemma_conversation_history.append({"role": role, "content": text})
        self.gemma_conversation_history = truncate_history(self.gemma_conversation_history)
        
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
                conversation_context=self.get_conversation_context()
            )
    
    def get_conversation_context(self) -> str:
        """Get conversation context for Gemma"""
        return format_conversation_context(self.gemma_conversation_history)
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.in_gemma_mode = False
        self.waiting_for_feedback = False
        self.gemma_conversation_history = []
        self.last_feedback = None