#!/usr/bin/env python3
"""Utility functions for the program pipeline"""

import re
from typing import List, Dict, Any

def is_question(text: str) -> bool:
    """Check if text is a question"""
    if '?' in text:
        return True
    
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
    text_lower = text.lower().strip()
    
    return (any(text_lower.startswith(word + ' ') for word in question_words) or
            any(text_lower.startswith(prefix) for prefix in ['is ', 'are ', 'do ', 'does ', 'can ', 'will ']))

def contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the specified keywords"""
    return any(keyword.lower() in text.lower() for keyword in keywords)

def truncate_history(history: List[Dict[str, Any]], max_items: int = 100) -> List[Dict[str, Any]]:
    """Keep only the last N items in history"""
    return history[-max_items:] if len(history) > max_items else history

def format_conversation_context(history: List[Dict[str, Any]], max_messages: int = 6) -> str:
    """Format conversation history for context"""
    if not history:
        return ""
    
    context = "Previous conversation:\n"
    for msg in history[-max_messages:]:
        context += f"{msg['role'].title()}: {msg['content']}\n"
    return context 