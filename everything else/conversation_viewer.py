#!/usr/bin/env python3
"""Conversation log viewer and analyzer"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from conversation_logger import list_conversation_sessions, get_conversation_history
import argparse

def view_sessions():
    """Display all conversation sessions"""
    sessions = list_conversation_sessions()
    
    if not sessions:
        print("📝 No conversation sessions found.")
        return
    
    print(f"📝 Found {len(sessions)} conversation session(s):\n")
    
    for i, session in enumerate(sessions, 1):
        start_time = session['start_time']
        end_time = session['end_time']
        duration = ""
        
        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration = end_dt - start_dt
            duration = f" ({duration})"
        
        print(f"{i}. {session['session_id']}")
        print(f"   📅 Start: {start_time}")
        print(f"   ⏱️  Duration: {duration}")
        print(f"   💬 Utterances: {session['total_utterances']}")
        print(f"   👥 Speakers: {session['speaker_count']}")
        print()

def view_session_detail(session_id: str):
    """Display detailed view of a specific session"""
    history = get_conversation_history(session_id)
    
    if not history['session']:
        print(f"❌ Session '{session_id}' not found.")
        return
    
    session = history['session']
    utterances = history['utterances']
    
    print(f"📝 Session: {session_id}")
    print("=" * 50)
    
    # Session info
    print(f"📅 Start Time: {session[2]}")
    print(f"⏱️  End Time: {session[3]}")
    print(f"💬 Total Utterances: {session[4]}")
    print(f"👥 Speaker Count: {session[5]}")
    print(f"😊 Emotion Summary: {session[6]}")
    print()
    
    # Conversation transcript
    print("🗣️  Conversation Transcript:")
    print("-" * 30)
    
    for utterance in utterances:
        timestamp = utterance[2]  # timestamp
        speaker = utterance[3]    # speaker
        text = utterance[4]       # text
        emotion = utterance[5]    # emotion
        confidence = utterance[6] # emotion_confidence
        is_question = utterance[7] # is_question
        
        question_mark = " [Q]" if is_question else ""
        print(f"[{timestamp}] {speaker}: {text} ({emotion}, {confidence:.2f}){question_mark}")
    
    print()

def view_session_json(session_id: str):
    """Display session data in JSON format"""
    storage_dir = Path("conversations")
    json_file = storage_dir / session_id / "conversation.json"
    
    if not json_file.exists():
        print(f"❌ JSON file for session '{session_id}' not found.")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(json.dumps(data, indent=2))

def analyze_emotions(session_id: str):
    """Analyze emotion patterns in a session"""
    history = get_conversation_history(session_id)
    
    if not history['utterances']:
        print(f"❌ No utterances found for session '{session_id}'.")
        return
    
    # Count emotions
    emotion_counts = {}
    question_count = 0
    total_utterances = len(history['utterances'])
    
    for utterance in history['utterances']:
        emotion = utterance[5]  # emotion column
        is_question = utterance[7]  # is_question column
        
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        if is_question:
            question_count += 1
    
    print(f"📊 Emotion Analysis for Session: {session_id}")
    print("=" * 40)
    print(f"💬 Total Utterances: {total_utterances}")
    print(f"❓ Questions: {question_count} ({question_count/total_utterances*100:.1f}%)")
    print()
    
    print("😊 Emotion Distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_utterances * 100
        print(f"   {emotion}: {count} ({percentage:.1f}%)")

def export_session(session_id: str, format: str = "json"):
    """Export session data to file"""
    if format == "json":
        storage_dir = Path("conversations")
        json_file = storage_dir / session_id / "conversation.json"
        
        if not json_file.exists():
            print(f"❌ JSON file for session '{session_id}' not found.")
            return
        
        # Copy to current directory
        output_file = f"{session_id}.json"
        import shutil
        shutil.copy2(json_file, output_file)
        print(f"📁 Exported session to: {output_file}")
    
    elif format == "txt":
        storage_dir = Path("conversations")
        summary_file = storage_dir / session_id / "summary.txt"
        
        if not summary_file.exists():
            print(f"❌ Summary file for session '{session_id}' not found.")
            return
        
        # Copy to current directory
        output_file = f"{session_id}.txt"
        import shutil
        shutil.copy2(summary_file, output_file)
        print(f"📁 Exported session to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Conversation log viewer and analyzer")
    parser.add_argument("command", choices=["list", "view", "json", "analyze", "export"], 
                       help="Command to execute")
    parser.add_argument("--session", "-s", help="Session ID for detailed operations")
    parser.add_argument("--format", "-f", choices=["json", "txt"], default="json",
                       help="Export format (default: json)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        view_sessions()
    
    elif args.command == "view":
        if not args.session:
            print("❌ Please specify a session ID with --session")
            return
        view_session_detail(args.session)
    
    elif args.command == "json":
        if not args.session:
            print("❌ Please specify a session ID with --session")
            return
        view_session_json(args.session)
    
    elif args.command == "analyze":
        if not args.session:
            print("❌ Please specify a session ID with --session")
            return
        analyze_emotions(args.session)
    
    elif args.command == "export":
        if not args.session:
            print("❌ Please specify a session ID with --session")
            return
        export_session(args.session, args.format)

if __name__ == "__main__":
    main() 