#!/usr/bin/env python3
"""Conversation database with audio features storage"""

import chromadb
import os
import json
import pickle
import numpy as np
import uuid
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
                                  latency_metrics: Optional[Dict] = None, model_used: Optional[str] = None):
        """Add conversation with audio features"""
        conversation_id = create_conversation_id(session_id)
        
        # Store conversation
        rich_text = f"{speaker} ({role}): {text}" + (" [GEMMA]" if is_gemma_mode else "")
        metadata = create_metadata(
            session_id, speaker, role, audio_features is not None,
            emotion_text=emotion_text, confidence=confidence,
            feedback=feedback, latency_metrics=latency_metrics,
            model_used=model_used
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
        """Get Gemma conversations for fine-tuning"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get all conversations
        data = self.conversations.get()
        
        gemma_conversations = []
        for i, metadata in enumerate(data['metadatas']):
            if metadata.get('is_gemma_mode') and metadata.get('session_id'):
                try:
                    timestamp = datetime.fromisoformat(metadata.get('timestamp', ''))
                    if timestamp >= cutoff_date:
                        gemma_conversations.append({
                            'text': data['documents'][i],
                            'metadata': metadata
                        })
                except:
                    continue
        
        return {
            'conversations': gemma_conversations,
            'total_count': len(gemma_conversations)
        }

    def get_vector_context(self, query: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """Get relevant vector context for a query"""
        try:
            # Search for cue cards
            cue_cards = self.search_cue_cards(query, top_k=top_k)
            
            # Search for adaptive prompts
            adaptive_prompts = self.search_adaptive_prompts(query, top_k=top_k)
            
            # Search for similar conversations
            similar_conversations = self.search_conversations(query, top_k=top_k)
            
            if not cue_cards and not adaptive_prompts and not similar_conversations:
                return None
            
            return {
                "relevant_cue_cards": cue_cards,
                "relevant_prompts": adaptive_prompts,
                "similar_conversations": similar_conversations
            }
        except Exception as e:
            print(f"Error getting vector context: {e}")
            return None

    def search_cue_cards(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for cue cards in the database"""
        try:
            # Build filter for cue cards
            filter_conditions = {"content_type": {"$eq": "cue_card"}}
            
            # Search the database
            results = self.conversations.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_conditions
            )
            
            # Format results
            cue_cards = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    cue_cards.append({
                        "question": metadata.get("question", ""),
                        "answer": metadata.get("answer", ""),
                        "prompt_type": metadata.get("prompt_type", ""),
                        "metadata": metadata
                    })
            
            return cue_cards
        except Exception as e:
            print(f"Error searching cue cards: {e}")
            return []

    def search_adaptive_prompts(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for adaptive prompts in the database"""
        try:
            # Build filter for adaptive prompts
            filter_conditions = {"content_type": {"$eq": "adaptive_prompt"}}
            
            # Search the database
            results = self.conversations.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_conditions
            )
            
            # Format results
            adaptive_prompts = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    adaptive_prompts.append({
                        "issue": metadata.get("medical_issue", ""),
                        "prompt": doc,
                        "metadata": metadata
                    })
            
            return adaptive_prompts
        except Exception as e:
            print(f"Error searching adaptive prompts: {e}")
            return []

    def search_conversations(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar conversations in the database"""
        try:
            # Search all conversations (excluding cue cards and adaptive prompts)
            results = self.conversations.query(
                query_texts=[query],
                n_results=top_k * 2  # Get more to filter out non-conversations
            )
            
            # Filter to actual conversations (not cue cards or adaptive prompts)
            conversations = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    
                    # Skip cue cards and adaptive prompts
                    if metadata.get('content_type') in ['cue_card', 'adaptive_prompt']:
                        continue
                    
                    conversations.append({
                        "text": doc,
                        "speaker": metadata.get("speaker", ""),
                        "role": metadata.get("role", ""),
                        "is_gemma_mode": metadata.get("is_gemma_mode", False),
                        "metadata": metadata
                    })
                    
                    if len(conversations) >= top_k:
                        break
            
            return conversations
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []

    def get_recent_conversations_with_feedback(self, days_back: int = 1) -> Dict[str, Dict]:
        """Get recent conversations with feedback for cue card updates"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get all conversations
            results = self.conversations.get(include=['metadatas', 'documents'])
            
            if not results.get('metadatas'):
                return {}
            
            # Group by session within date range
            sessions = {}
            for i, metadata in enumerate(results['metadatas']):
                timestamp_str = metadata.get('timestamp')
                if not timestamp_str:
                    continue
                
                try:
                    # Parse timestamp
                    conv_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if 'T' in timestamp_str else datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    
                    # Check if within date range and has Gemma content
                    if conv_time >= cutoff_date and metadata.get('is_gemma_mode'):
                        session_id = metadata.get('session_id')
                        if session_id not in sessions:
                            sessions[session_id] = []
                        
                        sessions[session_id].append({
                            'id': results['ids'][i],
                            'text': results['documents'][i],
                            'metadata': metadata,
                            'timestamp': conv_time,
                            'feedback_helpful': metadata.get('feedback_helpful', '')
                        })
                except Exception as e:
                    continue
            
            # Build result dictionary for sessions with feedback
            result_dict = {}
            for session_id, session_conversations in sessions.items():
                # Check if session has Gemma conversations
                has_gemma = any('[GEMMA]' in conv['text'] for conv in session_conversations)
                if not has_gemma:
                    continue
                
                # Sort by timestamp
                session_conversations.sort(key=lambda x: x['timestamp'])
                
                # Get session feedback
                session_feedback = None
                for conv in session_conversations:
                    if conv['feedback_helpful'] and conv['feedback_helpful'] != 'unknown':
                        session_feedback = conv['feedback_helpful']
                        break
                
                # Only include sessions with feedback
                if session_feedback:
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
                    
                    result_dict[session_id] = {
                        "session_feedback": session_feedback,
                        "full_text": "\n".join(conversation_lines),
                        "timestamp": session_conversations[0]['timestamp'],
                        "message_count": len(session_conversations),
                        "is_successful": session_feedback.lower() in ['yes', 'true', 'helpful', '1']
                    }
            
            return result_dict
        
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return {}

    def update_cue_card(self, cue_card_id: str, new_question: str, new_answer: str, update_reason: str = ""):
        """Update an existing cue card"""
        try:
            # Get the existing cue card
            data = self.conversations.get(ids=[cue_card_id])
            if not data['metadatas'] or not data['metadatas'][0]:
                return False
            
            metadata = data['metadatas'][0]
            
            # Update content and metadata
            new_content = f"Question: {new_question}\nAnswer: {new_answer}"
            updated_metadata = metadata.copy()
            updated_metadata['question'] = new_question
            updated_metadata['answer'] = new_answer
            updated_metadata['last_updated'] = datetime.now().isoformat()
            updated_metadata['update_reason'] = update_reason
            
            # Update in database
            self.conversations.update(
                ids=[cue_card_id],
                documents=[new_content],
                metadatas=[updated_metadata]
            )
            
            return True
        except Exception as e:
            print(f"Error updating cue card {cue_card_id}: {e}")
            return False

    def create_new_cue_card(self, question: str, answer: str, prompt_type: str, source: str = "conversation_update", model_used: str = "unknown"):
        """Create a new cue card based on conversation insights"""
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            content = f"Question: {question}\nAnswer: {answer}"
            metadata = {
                "document_path": f"conversation_update_{timestamp}",
                "cue_card_id": doc_id,
                "prompt_type": prompt_type,
                "question": question,
                "answer": answer,
                "timestamp": timestamp,
                "content_type": "cue_card",
                "session_id": f"update_session_{timestamp.replace(':', '-')}",
                "source": source,
                "created_from": "conversation_feedback",
                "model_used": model_used
            }
            
            # Store in database
            self.conversations.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"cue_card_{doc_id}"]
            )
            
            return f"cue_card_{doc_id}"
        except Exception as e:
            print(f"Error creating new cue card: {e}")
            return None