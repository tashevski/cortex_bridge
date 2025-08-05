#!/usr/bin/env python3
"""Database reset utility"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.enhanced_conversation_db import EnhancedConversationDB

def reset_database(reset_conversations=True, reset_audio_features=True):
    """Reset database collections selectively"""
    print("🔄 Resetting database...")
    
    db = EnhancedConversationDB()
    
    if reset_conversations:
        try:
            db.client.delete_collection("conversations")
            print("✅ Conversations collection deleted")
        except:
            print("ℹ️  Conversations collection didn't exist")
    
    if reset_audio_features:
        try:
            db.client.delete_collection("audio_features") 
            print("✅ Audio features collection deleted")
        except:
            print("ℹ️  Audio features collection didn't exist")
    
    print("🎉 Database reset complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reset database collections')
    parser.add_argument('--conversations-only', action='store_true', 
                       help='Reset only conversations')
    parser.add_argument('--audio-only', action='store_true',
                       help='Reset only audio features')
    
    args = parser.parse_args()
    
    if args.conversations_only:
        reset_database(reset_conversations=True, reset_audio_features=False)
    elif args.audio_only:
        reset_database(reset_conversations=False, reset_audio_features=True)
    else:
        reset_database()  # Reset everything