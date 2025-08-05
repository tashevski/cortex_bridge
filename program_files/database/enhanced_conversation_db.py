#!/usr/bin/env python3
"""Conversation database with audio features storage"""

import chromadb
import os
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .db_helpers import create_conversation_id, create_metadata, calculate_analytics, analyze_session

class EnhancedConversationDB:
    """Vector database with audio features storage"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            persist_directory = os.path.join(base_dir, "data", "vector_db")
        
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.conversations = self.client.get_or_create_collection("conversations")
        self.audio_features = self.client.get_or_create_collection("audio_features")
    
    def add_conversation_with_audio(self, session_id: str, text: str, speaker: str, 
                                  role: str, is_gemma_mode: bool, audio_features: Optional[Dict] = None,
                                  feedback: Optional[Dict] = None, conversation_context: Optional[str] = None,
                                  emotion_text: Optional[str] = None, confidence: Optional[float] = None,
                                  latency_metrics: Optional[Dict] = None):
        """Add conversation with audio features"""
        conversation_id = create_conversation_id(session_id)
        
        # Store conversation
        rich_text = f"{speaker} ({role}): {text}" + (" [GEMMA]" if is_gemma_mode else "")
        metadata = create_metadata(
            session_id, speaker, role, audio_features is not None,
            emotion_text=emotion_text, confidence=confidence,
            feedback=feedback, latency_metrics=latency_metrics
        )
        
        self.conversations.add(
            documents=[rich_text],
            metadatas=[metadata],
            ids=[conversation_id]
        )
        
        # Store audio features if available
        if audio_features:
            feature_doc = json.dumps({
                'features': [float(v) for v in audio_features.values()],
                'feature_names': list(audio_features.keys())
            })
            
            self.audio_features.add(
                documents=[feature_doc],
                metadatas=[{
                    'conversation_id': conversation_id,
                    'session_id': session_id,
                    'speaker': speaker,
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[f"audio_{conversation_id}"]
            )

    def update_session_with_feedback(self, session_id: str, feedback: Dict):
        """Update session messages with feedback"""
        data = self.conversations.get()
        
        for i, metadata in enumerate(data['metadatas']):
            if metadata.get('session_id') == session_id:
                updated_metadata = metadata.copy()
                updated_metadata['feedback_helpful'] = str(feedback.get('helpful', ''))
                
                self.conversations.update(
                    ids=[data['ids'][i]],
                    metadatas=[updated_metadata]
                )
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        conv_data = self.conversations.get(limit=1000)
        
        return {
            'total_conversations': self.conversations.count(),
            'total_audio_features': self.audio_features.count(),
            'conversations_with_audio': sum(1 for m in conv_data['metadatas'] if m.get('has_audio_features'))
        }

    # functions for reading and updating the database
    def get_data(self, collection="audio_features", return_features=True):
        """General function to get data from any collection"""
        target_collection = self.audio_features if collection == "audio_features" else self.conversations
        data = target_collection.get()
        
        if not data['documents']:
            return [], []
        
        if return_features and collection == "audio_features":
            # Parse features from documents
            features_list = []
            for doc in data['documents']:
                feature_data = json.loads(doc)
                features_list.append(feature_data['features'])
            return features_list, data['metadatas']
        
        return data['documents'], data['metadatas']
    
    def get_latency_analytics(self, session_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Get latency analytics from stored conversations"""
        # Query recent conversations with latency data
        where_clause = {"response_time": {"$gte": 0}}  # Has latency data
        if session_id:
            where_clause["session_id"] = session_id
            
        try:
            results = self.conversations.get(
                where=where_clause,
                limit=1000  # Get up to 1000 recent entries
            )
            
            if not results['metadatas']:
                return {"status": "no_data"}
            
            # Process latency metrics
            metrics = []
            for metadata in results['metadatas']:
                if 'response_time' in metadata:
                    metrics.append(metadata)
            
            if not metrics:
                return {"status": "no_latency_data"}
            
            return calculate_analytics(metrics)
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_problematic_sessions(self, interruption_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get sessions with high interruption rates or latency issues"""
        try:
            # Get all conversations with latency data
            results = self.conversations.get(
                where={"response_time": {"$gte": 0}},
                limit=1000
            )
            
            if not results['metadatas']:
                return []
            
            # Group by session
            sessions = {}
            for metadata in results['metadatas']:
                session_id = metadata.get('session_id', 'unknown')
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(metadata)
            
            # Analyze each session
            problematic_sessions = []
            for session_id, session_data in sessions.items():
                analysis = analyze_session(session_data)
                if analysis['interruption_rate'] >= interruption_threshold or analysis['high_latency_count'] >= analysis['total'] * 0.5:
                    problematic_sessions.append({
                        "session_id": session_id,
                        "total_responses": analysis['total'],
                        "interruption_rate": analysis['interruption_rate'],
                        "high_latency_count": analysis['high_latency_count'],
                        "timestamp": analysis['timestamp']
                    })
            
            return sorted(problematic_sessions, key=lambda x: x['interruption_rate'], reverse=True)
            
        except Exception as e:
            return []
        
    def update_by_indexes(self, updates_dict, collection="audio_features"):
        """General function to update database entries by index with field values"""
        target_collection = self.audio_features if collection == "audio_features" else self.conversations
        data = target_collection.get()
        
        if not data['metadatas']:
            return print(f"No data found in {collection}")
        
        updated_count = 0
        for index, field_updates in updates_dict.items():
            if index < len(data['metadatas']):
                current_meta = data['metadatas'][index]
                updated_meta = {**current_meta, **field_updates}
                
                target_collection.update(
                    ids=[data['ids'][index]],
                    metadatas=[updated_meta]
                )
                updated_count += 1
        
        return updated_count
    
    def get_conversations_by_date_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get conversations within a date range"""
        try:
            results = self.conversations.get(include=['metadatas'])
            if not results or not results.get('metadatas'): return []
            
            conversations = []
            for i, meta in enumerate(results['metadatas']):
                timestamp_str = meta.get('timestamp')
                if not timestamp_str: continue
                
                try:
                    conv_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if 'T' in timestamp_str else datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if start_time <= conv_time <= end_time:
                        conversations.append({'id': results['ids'][i], 'metadata': meta, 'timestamp': conv_time})
                except: continue
            return conversations
        except: return []
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation/session"""
        try:
            results = self.conversations.get(where={"session_id": conversation_id}, include=['metadatas', 'documents'])
            if not results or not results.get('metadatas'): return []
            
            return [{
                'id': results['ids'][i], 'text': doc, 'metadata': meta,
                'response_time': meta.get('response_time', 0), 'error': meta.get('error', False),
                'speaker_changed': meta.get('speaker_changed', False), 'interrupted': meta.get('interrupted', False),
                'model': meta.get('model', 'unknown')
            } for i, (meta, doc) in enumerate(zip(results['metadatas'], results.get('documents', [])))]
        except: return []
    
    def get_gemma_conversations_for_finetuning(self, days_back: int = 1) -> Dict[str, Any]:
        """Get Gemma conversations with feedback for fine-tuning, save as JSON"""
        results = self.conversations.get(include=['metadatas', 'documents'])
        if not results.get('metadatas'): return {}
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Group conversations by session within date range
        sessions = {}
        for i, metadata in enumerate(results['metadatas']):
            timestamp_str = metadata.get('timestamp')
            if not timestamp_str: continue
            
            try:
                conv_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if 'T' in timestamp_str else datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                if start_time <= conv_time <= end_time:
                    session_id = metadata.get('session_id')
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append({
                        'text': results['documents'][i],
                        'metadata': metadata,
                        'timestamp': conv_time,
                        'feedback_helpful': metadata.get('feedback_helpful', '')
                    })
            except: continue
        
        # Build result dictionary for sessions with Gemma
        result_dict = {}
        conversation_counter = 1
        
        for session_id, session_conversations in sessions.items():
            if not any('[GEMMA]' in conv['text'] for conv in session_conversations):
                continue
            
            session_conversations.sort(key=lambda x: x['timestamp'])
            
            # Get session feedback
            session_feedback = next((conv['feedback_helpful'] for conv in session_conversations 
                                   if conv['feedback_helpful'] and conv['feedback_helpful'] != 'unknown'), None)
            
            # Build conversation text
            conversation_lines = []
            for conv in session_conversations:
                full_text = conv['text']
                if ': ' in full_text:
                    speaker_text = full_text.split(': ', 1)[1]
                else:
                    speaker_text = full_text
                speaker_text = speaker_text.replace(' [GEMMA]', '')
                speaker = conv['metadata'].get('speaker', 'Unknown')
                conversation_lines.append(f"{speaker}: {speaker_text}")
            
            result_dict[f"conversation_{conversation_counter}"] = {
                "feedback_helpful": session_feedback,
                "full_text": "\n".join(conversation_lines),
                "session_id": session_id,
                "timestamp": session_conversations[0]['timestamp'].isoformat(),
                "message_count": len(session_conversations)
            }
            conversation_counter += 1
        
        # Save to JSON file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fine_tuning', 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"gemma_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        return result_dict