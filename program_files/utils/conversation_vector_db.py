#!/usr/bin/env python3
"""Conversation vector database using ChromaDB"""

import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional

class ConversationVectorDB:
    """Vector database for conversation storage and search"""
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Conversation embeddings with metadata"}
        )
    
    def add_conversation(self, session_id: str, text: str, speaker: str, 
                        role: str, is_gemma_mode: bool, feedback: Optional[Dict] = None,
                        conversation_context: Optional[str] = None):
        """Add a conversation to the vector database"""
        rich_text = self._create_rich_text(text, speaker, role, is_gemma_mode)
        
        metadata = {
            'session_id': session_id,
            'speaker': speaker,
            'role': role,
            'is_gemma_mode': is_gemma_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        if feedback and feedback.get('helpful') is not None:
            metadata['feedback_helpful'] = str(feedback['helpful'])
            metadata['feedback_response'] = feedback.get('response', '')
        
        if conversation_context:
            metadata['conversation_context'] = conversation_context
        
        self.collection.add(
            documents=[rich_text],
            metadatas=[metadata],
            ids=[f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"]
        )
    
    def update_session_with_feedback(self, session_id: str, feedback: Dict):
        """Update all messages in a session with feedback"""
        # Get all messages for this session
        all_data = self.collection.get()
        
        # Find indices of messages in this session
        session_indices = [
            i for i, metadata in enumerate(all_data['metadatas'])
            if metadata.get('session_id') == session_id
        ]
        
        if not session_indices:
            return
        
        # Update each message with feedback
        for idx in session_indices:
            metadata = all_data['metadatas'][idx].copy()
            metadata['feedback_helpful'] = str(feedback['helpful'])
            metadata['feedback_response'] = feedback.get('response', '')
            
            # Update the metadata in ChromaDB
            self.collection.update(
                ids=[all_data['ids'][idx]],
                metadatas=[metadata]
            )
    
    def search_conversations(self, query: str, top_k: int = 5, 
                           filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar conversations by individual messages"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )
        
        return [
            {'text': doc, 'metadata': metadata, 'distance': distance}
            for doc, metadata, distance in zip(
                results['documents'][0], results['metadatas'][0], results['distances'][0]
            )
        ]
    
    def search_by_conversation_context(self, query: str, top_k: int = 5,
                                     filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for conversations by their full context"""
        all_data = self.collection.get()
        
        if filter_metadata:
            filtered_indices = [
                i for i, metadata in enumerate(all_data['metadatas'])
                if all(metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        else:
            filtered_indices = list(range(len(all_data['metadatas'])))
        
        results = []
        for idx in filtered_indices:
            metadata = all_data['metadatas'][idx]
            context = metadata.get('conversation_context', '')
            
            if context and query.lower() in context.lower():
                similarity = len(set(query.lower().split()) & set(context.lower().split())) / len(set(query.lower().split()))
                results.append({
                    'session_id': metadata.get('session_id'),
                    'conversation_context': context,
                    'metadata': metadata,
                    'similarity': similarity
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        count = self.collection.count()
        sample = self.collection.get(limit=1000)
        
        gemma_count = sum(1 for m in sample['metadatas'] if m.get('is_gemma_mode'))
        user_count = sum(1 for m in sample['metadatas'] if m.get('role') == 'user')
        assistant_count = sum(1 for m in sample['metadatas'] if m.get('role') == 'assistant')
        context_count = sum(1 for m in sample['metadatas'] if m.get('conversation_context'))
        
        return {
            'total_conversations': count,
            'gemma_conversations': gemma_count,
            'user_messages': user_count,
            'assistant_messages': assistant_count,
            'messages_with_context': context_count
        }
    
    def _create_rich_text(self, text: str, speaker: str, role: str, 
                         is_gemma_mode: bool) -> str:
        """Create rich text representation for vectorization"""
        mode_marker = " [GEMMA]" if is_gemma_mode else ""
        return f"{speaker} ({role}): {text}{mode_marker}"