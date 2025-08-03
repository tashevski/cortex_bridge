#!/usr/bin/env python3
"""Conversation logging and storage system"""

import json
import os
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

class ConversationLogger:
    def __init__(self, storage_dir: str = "conversations"):
        """Initialize conversation logger with storage directory"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_dir / "conversations.db"
        self.init_database()
        
        # Current session
        self.current_session_id = None
        self.session_start_time = None
        
    def init_database(self):
        """Initialize SQLite database for conversation storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_utterances INTEGER DEFAULT 0,
                speaker_count INTEGER DEFAULT 0,
                emotion_summary TEXT
            )
        ''')
        
        # Create utterances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                speaker TEXT,
                text TEXT NOT NULL,
                emotion TEXT,
                emotion_confidence REAL,
                is_question BOOLEAN,
                voice_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create speaker_changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speaker_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                from_speaker TEXT,
                to_speaker TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        
        # Create session directory
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Initialize session in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sessions (session_id, start_time)
            VALUES (?, ?)
        ''', (session_id, self.session_start_time))
        conn.commit()
        conn.close()
        
        print(f"📝 Started conversation session: {session_id}")
        return session_id
    
    def log_utterance(self, text: str, speaker: str, emotion: str, 
                     emotion_confidence: float, is_question: bool, 
                     voice_count: int, timestamp: Optional[datetime] = None):
        """Log a single utterance with all metadata"""
        if self.current_session_id is None:
            self.start_session()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO utterances 
            (session_id, timestamp, speaker, text, emotion, emotion_confidence, is_question, voice_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.current_session_id, timestamp, speaker, text, emotion, 
              emotion_confidence, is_question, voice_count))
        
        # Update session stats
        cursor.execute('''
            UPDATE sessions 
            SET total_utterances = total_utterances + 1,
                speaker_count = ?
            WHERE session_id = ?
        ''', (voice_count, self.current_session_id))
        
        conn.commit()
        conn.close()
        
        # Also save to JSON file for easy access
        self.save_to_json(text, speaker, emotion, emotion_confidence, 
                         is_question, voice_count, timestamp)
    
    def log_speaker_change(self, from_speaker: str, to_speaker: str, 
                          timestamp: Optional[datetime] = None):
        """Log a speaker change event"""
        if self.current_session_id is None:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO speaker_changes (session_id, timestamp, from_speaker, to_speaker)
            VALUES (?, ?, ?, ?)
        ''', (self.current_session_id, timestamp, from_speaker, to_speaker))
        
        conn.commit()
        conn.close()
    
    def save_to_json(self, text: str, speaker: str, emotion: str, 
                    emotion_confidence: float, is_question: bool, 
                    voice_count: int, timestamp: datetime):
        """Save utterance to JSON file for easy reading"""
        session_dir = self.storage_dir / self.current_session_id
        json_file = session_dir / "conversation.json"
        
        utterance_data = {
            "timestamp": timestamp.isoformat(),
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "is_question": is_question,
            "voice_count": voice_count
        }
        
        # Load existing data or create new
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"session_id": self.current_session_id, "utterances": []}
        
        data["utterances"].append(utterance_data)
        
        # Save updated data
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def end_session(self):
        """End the current conversation session"""
        if self.current_session_id is None:
            return
        
        end_time = datetime.now()
        
        # Update session end time and generate summary
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get emotion summary
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM utterances 
            WHERE session_id = ? 
            GROUP BY emotion 
            ORDER BY count DESC
        ''', (self.current_session_id,))
        
        emotion_counts = cursor.fetchall()
        emotion_summary = ", ".join([f"{emotion}: {count}" for emotion, count in emotion_counts])
        
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, emotion_summary = ?
            WHERE session_id = ?
        ''', (end_time, emotion_summary, self.current_session_id))
        
        conn.commit()
        conn.close()
        
        # Create session summary
        self.create_session_summary()
        
        print(f"📝 Ended conversation session: {self.current_session_id}")
        print(f"⏱️  Duration: {end_time - self.session_start_time}")
        
        self.current_session_id = None
        self.session_start_time = None
    
    def create_session_summary(self):
        """Create a summary file for the session"""
        if self.current_session_id is None:
            return
        
        session_dir = self.storage_dir / self.current_session_id
        summary_file = session_dir / "summary.txt"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('''
            SELECT start_time, end_time, total_utterances, speaker_count, emotion_summary
            FROM sessions WHERE session_id = ?
        ''', (self.current_session_id,))
        
        session_info = cursor.fetchone()
        
        # Get all utterances
        cursor.execute('''
            SELECT timestamp, speaker, text, emotion, is_question
            FROM utterances WHERE session_id = ?
            ORDER BY timestamp
        ''', (self.current_session_id,))
        
        utterances = cursor.fetchall()
        
        conn.close()
        
        # Create summary
        with open(summary_file, 'w') as f:
            f.write(f"Conversation Session: {self.current_session_id}\n")
            f.write("=" * 50 + "\n\n")
            
            if session_info:
                start_time, end_time, total_utterances, speaker_count, emotion_summary = session_info
                f.write(f"Start Time: {start_time}\n")
                f.write(f"End Time: {end_time}\n")
                f.write(f"Total Utterances: {total_utterances}\n")
                f.write(f"Speaker Count: {speaker_count}\n")
                f.write(f"Emotion Summary: {emotion_summary}\n\n")
            
            f.write("Conversation Transcript:\n")
            f.write("-" * 30 + "\n")
            
            for timestamp, speaker, text, emotion, is_question in utterances:
                question_mark = " [Q]" if is_question else ""
                f.write(f"[{timestamp}] {speaker}: {text} ({emotion}){question_mark}\n")
    
    def get_session_history(self, session_id: str) -> Dict:
        """Retrieve conversation history for a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('''
            SELECT * FROM sessions WHERE session_id = ?
        ''', (session_id,))
        
        session = cursor.fetchone()
        
        # Get utterances
        cursor.execute('''
            SELECT * FROM utterances WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        utterances = cursor.fetchall()
        
        conn.close()
        
        return {
            "session": session,
            "utterances": utterances
        }
    
    def list_sessions(self) -> List[Dict]:
        """List all available conversation sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, start_time, end_time, total_utterances, speaker_count
            FROM sessions ORDER BY start_time DESC
        ''')
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": session[0],
                "start_time": session[1],
                "end_time": session[2],
                "total_utterances": session[3],
                "speaker_count": session[4]
            }
            for session in sessions
        ]

# Global logger instance
conversation_logger = ConversationLogger()

def log_conversation(text: str, speaker: str, emotion: str, 
                    emotion_confidence: float, is_question: bool, 
                    voice_count: int):
    """Convenience function to log conversation data"""
    conversation_logger.log_utterance(
        text, speaker, emotion, emotion_confidence, is_question, voice_count
    )

def start_conversation_session(session_id: Optional[str] = None) -> str:
    """Start a new conversation session"""
    return conversation_logger.start_session(session_id)

def end_conversation_session():
    """End the current conversation session"""
    conversation_logger.end_session()

def get_conversation_history(session_id: str) -> Dict:
    """Get conversation history for a session"""
    return conversation_logger.get_session_history(session_id)

def list_conversation_sessions() -> List[Dict]:
    """List all conversation sessions"""
    return conversation_logger.list_sessions() 