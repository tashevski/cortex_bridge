#!/usr/bin/env python3
"""Conversation database with audio features storage"""

import chromadb
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

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
                                  feedback: Optional[Dict] = None, conversation_context: Optional[str] = None):
        """Add conversation with audio features"""
        conversation_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store conversation
        rich_text = f"{speaker} ({role}): {text}" + (" [GEMMA]" if is_gemma_mode else "")
        metadata = {
            'session_id': session_id,
            'speaker': speaker,
            'role': role,
            'timestamp': datetime.now().isoformat(),
            'has_audio_features': audio_features is not None
        }
        
        if feedback:
            metadata['feedback_helpful'] = str(feedback.get('helpful', ''))
        
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