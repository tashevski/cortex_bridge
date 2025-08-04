#!/usr/bin/env python3
"""Test conversation mode configuration"""

import sys
sys.path.append('program_files')

from utils.config import ConversationModeConfig, cfg
from core.conversation_manager import ConversationManager

def test_conversation_mode_config():
    """Test different conversation mode configurations"""
    
    print("🎯 Testing Conversation Mode Configuration")
    print("=" * 60)
    
    # Test 1: Default configuration
    print("\n📝 Test 1: Default Configuration")
    manager = ConversationManager(enable_vector_db=False)
    
    test_inputs = [
        ("What time is it?", "Question with question mark"),
        ("How are you", "Question without question mark"),
        ("Hey Gemma", "Enter keyword"),
        ("Is it raining", "Auxiliary verb question"),
        ("Hello there", "Regular statement"),
        ("Exit conversation", "Exit keyword")
    ]
    
    for text, description in test_inputs:
        is_question = manager.is_question(text)
        should_enter = manager.should_enter_gemma_mode(text)
        should_exit = manager.should_exit_gemma_mode(text)
        print(f"  '{text}' ({description})")
        print(f"    Question: {is_question}, Enter: {should_enter}, Exit: {should_exit}")
    
    # Test 2: Disable question auto-entry
    print("\n📝 Test 2: Disable Question Auto-Entry")
    no_question_config = ConversationModeConfig(
        enter_on_questions=False  # Only keywords trigger entry
    )
    no_question_manager = ConversationManager(enable_vector_db=False, config=no_question_config)
    
    for text, description in test_inputs:
        should_enter = no_question_manager.should_enter_gemma_mode(text)
        print(f"  '{text}': Enter: {should_enter}")
    
    # Test 3: Custom keywords
    print("\n📝 Test 3: Custom Keywords")
    custom_config = ConversationModeConfig(
        enter_keywords=['jarvis', 'computer', 'ai assistant'],
        exit_keywords=['dismiss', 'that will be all'],
        question_words=['tell', 'show', 'explain'],  # Custom question words
        auxiliary_prefixes=['could ', 'would ', 'should ']
    )
    custom_manager = ConversationManager(enable_vector_db=False, config=custom_config)
    
    custom_test_inputs = [
        ("Computer, what time is it?", "Custom enter keyword"),
        ("Tell me about AI", "Custom question word"),
        ("Could you help me", "Custom auxiliary prefix"),
        ("That will be all", "Custom exit keyword"),
        ("What are you doing", "Standard question word - should not trigger")
    ]
    
    for text, description in custom_test_inputs:
        is_question = custom_manager.is_question(text)
        should_enter = custom_manager.should_enter_gemma_mode(text)
        should_exit = custom_manager.should_exit_gemma_mode(text)
        print(f"  '{text}' ({description})")
        print(f"    Question: {is_question}, Enter: {should_enter}, Exit: {should_exit}")

def show_configuration_examples():
    """Show different configuration examples"""
    
    print("\n🔧 Configuration Examples")
    print("=" * 60)
    
    # Voice assistant mode
    print("\n🎤 Voice Assistant Mode (like Alexa/Siri):")
    voice_config = ConversationModeConfig(
        enter_keywords=['hey computer', 'computer'],
        exit_keywords=['stop', 'cancel'],
        enter_on_questions=False  # Only respond when called
    )
    print(f"  Enter keywords: {voice_config.enter_keywords}")
    print(f"  Auto-enter on questions: {voice_config.enter_on_questions}")
    
    # Chat mode
    print("\n💬 Chat Mode (always responsive):")
    chat_config = ConversationModeConfig(
        enter_keywords=['hi', 'hello'],
        exit_keywords=['bye', 'goodbye'],
        enter_on_questions=True  # Any question starts conversation
    )
    print(f"  Enter on questions: {chat_config.enter_on_questions}")
    
    # Professional mode
    print("\n👔 Professional Mode (formal language):")
    professional_config = ConversationModeConfig(
        enter_keywords=['assistant', 'please help'],
        exit_keywords=['thank you', 'that will be all'],
        question_words=['could you', 'would you', 'please'],
        auxiliary_prefixes=['could ', 'would ', 'might ']
    )
    print(f"  Question words: {professional_config.question_words}")
    print(f"  Auxiliary prefixes: {professional_config.auxiliary_prefixes}")
    
    # Research mode
    print("\n🔬 Research Mode (specific triggers):")
    research_config = ConversationModeConfig(
        enter_keywords=['research assistant', 'help me research'],
        exit_keywords=['end research', 'stop research'],
        question_words=['find', 'search', 'lookup', 'research']
    )
    print(f"  Enter keywords: {research_config.enter_keywords}")
    print(f"  Custom question words: {research_config.question_words}")

if __name__ == "__main__":
    test_conversation_mode_config()
    show_configuration_examples()