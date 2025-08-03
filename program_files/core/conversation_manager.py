#!/usr/bin/env python3
"""Conversation management and state handling"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.utils import is_question, contains_keywords, truncate_history, format_conversation_context
from utils.conversation_vector_db import ConversationVectorDB

class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, enable_vector_db: bool = True):
        self.in_gemma_mode = False
        self.waiting_for_feedback = False
        self.gemma_conversation_history = []
        self.exit_keywords = ['exit', 'quit', 'stop', 'bye', 'goodbye', 'end conversation']
        self.enter_keywords = ['hey gemma', 'gemma', 'ai', 'assistant', 'help']
        self.last_feedback = None
        self.session_id = self._generate_session_id()
        
        # Initialize vector database
        self.vector_db = ConversationVectorDB() if enable_vector_db else None
    
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
        return (contains_keywords(text_lower, self.enter_keywords) or 
                is_question(text))
    
    def should_exit_gemma_mode(self, text: str) -> bool:
        """Check if we should exit Gemma conversation mode"""
        return contains_keywords(text.lower().strip(), self.exit_keywords)
    
    def add_to_history(self, text: str, is_user: bool = True, speaker_name: str = None):
        """Add message to conversation history and vector database"""
        role = "user" if is_user else "assistant"
        self.gemma_conversation_history.append({"role": role, "content": text})
        self.gemma_conversation_history = truncate_history(self.gemma_conversation_history)
        
        # Store in vector database with conversation context
        if self.vector_db:
            self.vector_db.add_conversation(
                session_id=self.session_id,
                text=text,
                speaker=speaker_name or "Unknown",
                role=role,
                is_gemma_mode=self.in_gemma_mode,
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