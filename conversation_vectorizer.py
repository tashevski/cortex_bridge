#!/usr/bin/env python3
"""Conversation data vectorization for semantic search and analysis"""

import json
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from conversation_logger import list_conversation_sessions, get_conversation_history
import argparse

class ConversationVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dir: str = "vectors"):
        """Initialize vectorizer with sentence transformer model"""
        self.model_name = model_name
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(exist_ok=True)
        
        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Vector storage
        self.vectors_file = self.vector_dir / "conversation_vectors.pkl"
        self.metadata_file = self.vector_dir / "conversation_metadata.pkl"
        self.vectors = []
        self.metadata = []
        
        # Load existing vectors if available
        self.load_vectors()
    
    def load_vectors(self):
        """Load existing vectors from disk"""
        if self.vectors_file.exists() and self.metadata_file.exists():
            try:
                with open(self.vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded {len(self.vectors)} existing vectors")
            except Exception as e:
                print(f"Error loading vectors: {e}")
                self.vectors = []
                self.metadata = []
    
    def save_vectors(self):
        """Save vectors to disk"""
        with open(self.vectors_file, 'wb') as f:
            pickle.dump(self.vectors, f)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved {len(self.vectors)} vectors to disk")
    
    def create_text_for_vectorization(self, utterance_data: Dict) -> str:
        """Create text representation for vectorization"""
        text = utterance_data['text']
        speaker = utterance_data['speaker']
        emotion = utterance_data['emotion']
        is_question = utterance_data['is_question']
        
        # Create rich text representation
        question_marker = " [QUESTION]" if is_question else ""
        emotion_marker = f" [EMOTION: {emotion}]" if emotion else ""
        
        return f"{speaker}: {text}{question_marker}{emotion_marker}"
    
    def vectorize_session(self, session_id: str, force_recompute: bool = False) -> int:
        """Vectorize all utterances in a session"""
        if not force_recompute and self.session_already_vectorized(session_id):
            print(f"Session {session_id} already vectorized, skipping...")
            return 0
        
        # Get session history
        history = get_conversation_history(session_id)
        if not history['utterances']:
            print(f"No utterances found for session {session_id}")
            return 0
        
        # Remove existing vectors for this session
        self.remove_session_vectors(session_id)
        
        # Prepare texts for vectorization
        texts = []
        session_metadata = []
        
        for utterance in history['utterances']:
            # Create utterance data dict
            utterance_data = {
                'session_id': session_id,
                'timestamp': utterance[2],  # timestamp
                'speaker': utterance[3],    # speaker
                'text': utterance[4],       # text
                'emotion': utterance[5],    # emotion
                'emotion_confidence': utterance[6], # emotion_confidence
                'is_question': utterance[7], # is_question
                'voice_count': utterance[8]  # voice_count
            }
            
            # Create text for vectorization
            text = self.create_text_for_vectorization(utterance_data)
            texts.append(text)
            session_metadata.append(utterance_data)
        
        # Vectorize texts
        print(f"Vectorizing {len(texts)} utterances from session {session_id}")
        vectors = self.model.encode(texts, show_progress_bar=True)
        
        # Store vectors and metadata
        self.vectors.extend(vectors)
        self.metadata.extend(session_metadata)
        
        print(f"Added {len(vectors)} vectors for session {session_id}")
        return len(vectors)
    
    def vectorize_all_sessions(self, force_recompute: bool = False) -> int:
        """Vectorize all available sessions"""
        sessions = list_conversation_sessions()
        total_vectors = 0
        
        print(f"Vectorizing {len(sessions)} sessions...")
        
        for session in sessions:
            session_id = session['session_id']
            vectors_added = self.vectorize_session(session_id, force_recompute)
            total_vectors += vectors_added
        
        # Save all vectors
        self.save_vectors()
        
        print(f"Total vectors created: {total_vectors}")
        return total_vectors
    
    def session_already_vectorized(self, session_id: str) -> bool:
        """Check if a session is already vectorized"""
        return any(meta['session_id'] == session_id for meta in self.metadata)
    
    def remove_session_vectors(self, session_id: str):
        """Remove vectors for a specific session"""
        indices_to_remove = []
        for i, meta in enumerate(self.metadata):
            if meta['session_id'] == session_id:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.vectors[i]
            del self.metadata[i]
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search on vectorized conversations"""
        if not self.vectors:
            print("No vectors available for search")
            return []
        
        # Vectorize query
        query_vector = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            metadata = self.metadata[idx]
            
            results.append({
                'similarity': float(similarity),
                'session_id': metadata['session_id'],
                'timestamp': metadata['timestamp'],
                'speaker': metadata['speaker'],
                'text': metadata['text'],
                'emotion': metadata['emotion'],
                'is_question': metadata['is_question']
            })
        
        return results
    
    def find_similar_utterances(self, utterance_text: str, top_k: int = 5) -> List[Dict]:
        """Find utterances similar to a given text"""
        return self.semantic_search(utterance_text, top_k)
    
    def emotion_analysis_search(self, emotion: str, top_k: int = 10) -> List[Dict]:
        """Find utterances with specific emotions"""
        if not self.vectors:
            return []
        
        # Filter by emotion
        emotion_utterances = []
        for i, meta in enumerate(self.metadata):
            if meta['emotion'].lower() == emotion.lower():
                emotion_utterances.append((i, meta))
        
        # Sort by emotion confidence
        emotion_utterances.sort(key=lambda x: x[1]['emotion_confidence'], reverse=True)
        
        results = []
        for idx, meta in emotion_utterances[:top_k]:
            results.append({
                'confidence': meta['emotion_confidence'],
                'session_id': meta['session_id'],
                'timestamp': meta['timestamp'],
                'speaker': meta['speaker'],
                'text': meta['text'],
                'emotion': meta['emotion'],
                'is_question': meta['is_question']
            })
        
        return results
    
    def speaker_analysis(self, speaker: str, top_k: int = 10) -> List[Dict]:
        """Find utterances from a specific speaker"""
        if not self.vectors:
            return []
        
        speaker_utterances = []
        for i, meta in enumerate(self.metadata):
            if meta['speaker'].lower() == speaker.lower():
                speaker_utterances.append((i, meta))
        
        # Sort by timestamp
        speaker_utterances.sort(key=lambda x: x[1]['timestamp'])
        
        results = []
        for idx, meta in speaker_utterances[:top_k]:
            results.append({
                'session_id': meta['session_id'],
                'timestamp': meta['timestamp'],
                'speaker': meta['speaker'],
                'text': meta['text'],
                'emotion': meta['emotion'],
                'emotion_confidence': meta['emotion_confidence'],
                'is_question': meta['is_question']
            })
        
        return results
    
    def get_vector_stats(self) -> Dict:
        """Get statistics about vectorized data"""
        if not self.vectors:
            return {"total_vectors": 0}
        
        # Count by session
        session_counts = {}
        emotion_counts = {}
        speaker_counts = {}
        question_count = 0
        
        for meta in self.metadata:
            session_id = meta['session_id']
            emotion = meta['emotion']
            speaker = meta['speaker']
            
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            if meta['is_question']:
                question_count += 1
        
        return {
            "total_vectors": len(self.vectors),
            "unique_sessions": len(session_counts),
            "unique_speakers": len(speaker_counts),
            "unique_emotions": len(emotion_counts),
            "questions": question_count,
            "session_counts": session_counts,
            "emotion_counts": emotion_counts,
            "speaker_counts": speaker_counts
        }

def main():
    parser = argparse.ArgumentParser(description="Conversation data vectorization and semantic search")
    parser.add_argument("command", choices=["vectorize", "search", "emotion", "speaker", "stats"], 
                       help="Command to execute")
    parser.add_argument("--session", "-s", help="Session ID for vectorization")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--emotion", "-e", help="Emotion to search for")
    parser.add_argument("--speaker", "-p", help="Speaker to search for")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--force", "-f", action="store_true", help="Force recomputation of vectors")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    
    args = parser.parse_args()
    
    vectorizer = ConversationVectorizer(model_name=args.model)
    
    if args.command == "vectorize":
        if args.session:
            vectorizer.vectorize_session(args.session, args.force)
        else:
            vectorizer.vectorize_all_sessions(args.force)
    
    elif args.command == "search":
        if not args.query:
            print("❌ Please specify a search query with --query")
            return
        
        results = vectorizer.semantic_search(args.query, args.top_k)
        print(f"\n🔍 Semantic search results for: '{args.query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Similarity: {result['similarity']:.3f}")
            print(f"   Session: {result['session_id']}")
            print(f"   Speaker: {result['speaker']}")
            print(f"   Text: {result['text']}")
            print(f"   Emotion: {result['emotion']}")
            print()
    
    elif args.command == "emotion":
        if not args.emotion:
            print("❌ Please specify an emotion with --emotion")
            return
        
        results = vectorizer.emotion_analysis_search(args.emotion, args.top_k)
        print(f"\n😊 Emotion search results for: '{args.emotion}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Confidence: {result['confidence']:.3f}")
            print(f"   Session: {result['session_id']}")
            print(f"   Speaker: {result['speaker']}")
            print(f"   Text: {result['text']}")
            print()
    
    elif args.command == "speaker":
        if not args.speaker:
            print("❌ Please specify a speaker with --speaker")
            return
        
        results = vectorizer.speaker_analysis(args.speaker, args.top_k)
        print(f"\n👤 Speaker analysis for: '{args.speaker}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Session: {result['session_id']}")
            print(f"   Time: {result['timestamp']}")
            print(f"   Text: {result['text']}")
            print(f"   Emotion: {result['emotion']} ({result['emotion_confidence']:.3f})")
            print()
    
    elif args.command == "stats":
        stats = vectorizer.get_vector_stats()
        print("\n📊 Vectorization Statistics")
        print("=" * 40)
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Unique sessions: {stats['unique_sessions']}")
        print(f"Unique speakers: {stats['unique_speakers']}")
        print(f"Unique emotions: {stats['unique_emotions']}")
        print(f"Questions: {stats['questions']}")
        
        if stats['emotion_counts']:
            print("\n😊 Emotion distribution:")
            for emotion, count in sorted(stats['emotion_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {emotion}: {count}")
        
        if stats['speaker_counts']:
            print("\n👥 Speaker distribution:")
            for speaker, count in sorted(stats['speaker_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {speaker}: {count}")

if __name__ == "__main__":
    main() 