import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_functions.utils.ocr_layout_copy import extract_text_and_layout
from rag_functions.utils.semantic_parser import parse_document
from rag_functions.utils.retrieval import setup_vector_db, retrieve_references, extract_medical_issues_list
from rag_functions.core.llm_analysis import analyze_with_llm, create_cue_cards
from rag_functions.templates.prompt_templates import get_template
# Optional ML imports (require sentence_transformers)
try:
    from rag_functions.ml.vector_operations import select_optimal_templates, analyze_document_type
    from rag_functions.ml.cue_card_extraction import extract_cue_cards
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available: {e}")
    ML_AVAILABLE = False
    # Provide stub functions
    def select_optimal_templates(*args, **kwargs):
        return []
    def analyze_document_type(*args, **kwargs):
        return {}
    def extract_cue_cards(*args, **kwargs):
        return []
from rag_functions.core.config import get_config
from program_files.ai.gemma_client import GemmaClient
from program_files.database.enhanced_conversation_db import EnhancedConversationDB
import re
import json
import uuid
from datetime import datetime


def setup_rag_vector_db():
    """Setup vector database for RAG functions"""
    # Use the same database as the main program
    base_dir = Path(__file__).parent.parent.parent / "program_files"
    persist_directory = base_dir / "data" / "vector_db"
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    return EnhancedConversationDB(str(persist_directory))


def store_cue_cards_in_db(vector_db, document_path, cue_cards, prompt_type, model_used: str = "unknown"):
    """Store cue cards in the vector database"""
    if not cue_cards or isinstance(cue_cards, dict) and "error" in cue_cards:
        return
    
    # Create a unique ID for this document's cue cards
    doc_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Store each cue card as a separate document
    for i, (key, value) in enumerate(cue_cards.items()):
        if isinstance(value, dict) and "question" in value and "answer" in value:
            # Format the cue card content
            content = f"Question: {value['question']}\nAnswer: {value['answer']}"
            
            # Create metadata
            metadata = {
                "document_path": str(document_path),
                "cue_card_id": f"{doc_id}_{i}",
                "prompt_type": prompt_type,
                "question": value['question'],
                "answer": value['answer'],
                "timestamp": timestamp,
                "content_type": "cue_card",
                "session_id": f"rag_session_{timestamp.replace(':', '-')}",
                "model_used": model_used
            }
            
            # Store in vector database
            vector_db.conversations.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"cue_card_{doc_id}_{i}"]
            )
            print(f"✓ Stored cue card {i+1} for {prompt_type}")


def store_adaptive_prompts_in_db(vector_db, document_path, adaptive_prompts, medical_issues):
    """Store adaptive prompts in the vector database"""
    if not adaptive_prompts:
        return
    
    timestamp = datetime.now().isoformat()
    
    # Store each adaptive prompt with its corresponding medical issue
    for i, (prompt, issue) in enumerate(zip(adaptive_prompts, medical_issues)):
        # Create metadata
        metadata = {
            "document_path": str(document_path),
            "prompt_id": f"adaptive_{uuid.uuid4()}",
            "medical_issue": issue,
            "prompt_text": prompt,
            "timestamp": timestamp,
            "content_type": "adaptive_prompt",
            "session_id": f"rag_session_{timestamp.replace(':', '-')}"
        }
        
        # Store in vector database
        vector_db.conversations.add(
            documents=[prompt],
            metadatas=[metadata],
            ids=[f"adaptive_prompt_{i}_{timestamp.replace(':', '-')}"]
        )
        print(f"✓ Stored adaptive prompt for issue: {issue}")


def process_document(file_path, reference_texts=None, use_medical_templates=True, generate_cue_cards=True, context_type="medical"):
    config = get_config()
    
    # Setup vector database for RAG functions
    vector_db = setup_rag_vector_db()
    
    # Extract and parse
    txt = extract_text_and_layout(file_path)
    parsed = parse_document(txt, input_prompt="Extract key entities, topics, and sections from the following document. Provide a structured summary:")
    
    client = GemmaClient(model="gemma3n:e4b")
    model_used = client.model
    key_medical_issues_response = client.generate_response("""identify the key medical concerns and likely problems the patient is likely to face given the information provided
        return your responses in a list of strings like this:
        [
            "{medical_concern}",
            "{medical_concern}",
            "{medical_concern}"
        ]
        """, parsed, timeout=90)

    # Extract the list from the response
    key_medical_issues = extract_medical_issues_list(key_medical_issues_response)

    adaptive_prompts = []
    for issue in key_medical_issues:
        prompt = f"briefly summarise and identify any issues relating to {issue} in the associated conversations, briefly describe what happened and whether it was effective:"
        print(prompt)
        adaptive_prompts.append(prompt)
    
    # Store adaptive prompts in vector database
    store_adaptive_prompts_in_db(vector_db, file_path, adaptive_prompts, key_medical_issues)
    
    individual_relevant_prompts = [
        "medical and care advice for family",
        "medical and care advice for medical staff",
        "medical and care advice for carers",
        "medical and care advice for allied health workers relevant to the context",
        "medical and care advice for doctors relevant to the context",
        ]
   
    # Process each prompt separately and combine responses
    contextual_responses = {}
    for prompt in individual_relevant_prompts:
        contextual_responses[prompt] = create_cue_cards(txt, prompt)
        # Store cue cards in vector database
        store_cue_cards_in_db(vector_db, file_path, contextual_responses[prompt], prompt, model_used)
        #print(contextual_responses[prompt])

    # Setup references
    references = []
    if reference_texts:
        vectorstore = setup_vector_db(reference_texts, None)
        references = retrieve_references(vectorstore, parsed, k=config.max_reference_chunks)
    
    print(f"\n✓ Document processing complete!")
    print(f"✓ Stored {len(adaptive_prompts)} adaptive prompts")
    print(f"✓ Stored cue cards for {len(contextual_responses)} prompt types")
    
    return {
        "adaptive_prompts": adaptive_prompts,
        "contextual_responses": contextual_responses,
        "medical_issues": key_medical_issues,
        "references": references
    }


def update_cue_cards_from_conversations(days_back: int = 1, similarity_threshold: float = 0.7):
    """Update cue cards based on recent conversation feedback"""
    print(f"🔄 Updating cue cards from last {days_back} day(s) of conversations...")
    
    # Setup
    vector_db = setup_rag_vector_db()
    client = GemmaClient(model="gemma3n:e4b")
    model_used = client.model
    
    # Get recent conversations with feedback
    recent_conversations = vector_db.get_recent_conversations_with_feedback(days_back)
    
    if not recent_conversations:
        print("   No recent conversations with feedback found")
        return
    
    print(f"   Found {len(recent_conversations)} conversations with feedback")
    
    # Process each conversation
    updates_made = 0
    new_cards_created = 0
    
    for session_id, conv_data in recent_conversations.items():
        conversation_text = conv_data['full_text']
        is_successful = conv_data['is_successful']
        feedback = conv_data['session_feedback']
        
        print(f"\n📝 Processing conversation: {session_id}")
        print(f"   Feedback: {feedback} ({'✅ Successful' if is_successful else '❌ Unsuccessful'})")
        
        # Find similar cue cards using vector search
        similar_cue_cards = vector_db.search_cue_cards(conversation_text, top_k=5)
        
        if similar_cue_cards:
            # Check similarity and update existing cards
            for cue_card in similar_cue_cards:
                card_content = f"{cue_card['question']} {cue_card['answer']}"
                
                # Use vector similarity to determine if update is needed
                similarity_check_prompt = f"""
                Compare these two medical conversations and determine if they address the same medical topic/issue.
                
                Existing cue card:
                Q: {cue_card['question']}
                A: {cue_card['answer']}
                
                Recent conversation:
                {conversation_text}
                
                Return only "SIMILAR" if they address the same medical topic, or "DIFFERENT" if they don't.
                """
                
                try:
                    similarity_result = client.generate_response(similarity_check_prompt, "", timeout=30)
                    is_similar = "SIMILAR" in similarity_result.upper()
                    
                    if is_similar:
                        print(f"   🔍 Found similar cue card: {cue_card['question'][:50]}...")
                        
                        # Generate updated cue card based on conversation outcome
                        update_prompt = f"""
                        Update this medical cue card based on a recent conversation and its outcome.
                        
                        Current cue card:
                        Q: {cue_card['question']}
                        A: {cue_card['answer']}
                        
                        Recent conversation:
                        {conversation_text}
                        
                        Conversation outcome: {'Successful/Helpful' if is_successful else 'Unsuccessful/Not helpful'}
                        
                        Based on this feedback, provide an improved cue card in this exact format:
                        QUESTION: [improved question]
                        ANSWER: [improved answer incorporating lessons learned]
                        
                        If the conversation was successful, reinforce what worked.
                        If unsuccessful, adjust the advice to address what didn't work.
                        """
                        
                        try:
                            update_response = client.generate_response(update_prompt, "", timeout=60)
                            
                            # Extract question and answer
                            question_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:|$)', update_response, re.DOTALL | re.IGNORECASE)
                            answer_match = re.search(r'ANSWER:\s*(.+?)$', update_response, re.DOTALL | re.IGNORECASE)
                            
                            if question_match and answer_match:
                                new_question = question_match.group(1).strip()
                                new_answer = answer_match.group(1).strip()
                                
                                # Update the cue card
                                update_reason = f"Updated based on {'successful' if is_successful else 'unsuccessful'} conversation feedback"
                                success = vector_db.update_cue_card(
                                    cue_card['metadata']['id'], 
                                    new_question, 
                                    new_answer, 
                                    update_reason
                                )
                                
                                if success:
                                    print(f"   ✅ Updated cue card: {new_question[:50]}...")
                                    updates_made += 1
                                else:
                                    print(f"   ❌ Failed to update cue card")
                            else:
                                print(f"   ❌ Could not parse update response")
                                
                        except Exception as e:
                            print(f"   ❌ Error generating update: {e}")
                        
                        break  # Only update one similar card per conversation
                        
                except Exception as e:
                    print(f"   ❌ Error checking similarity: {e}")
        
        else:
            # No similar cue cards found - create new one if conversation has new insights
            print(f"   🆕 No similar cue cards found - checking for new insights...")
            
            new_card_prompt = f"""
            Analyze this medical conversation and determine if it contains valuable medical insights that should be captured as a cue card.
            
            Conversation:
            {conversation_text}
            
            Conversation outcome: {'Successful/Helpful' if is_successful else 'Unsuccessful/Not helpful'}
            
            If this conversation contains valuable medical advice or insights, create a cue card in this format:
            QUESTION: [relevant medical question]
            ANSWER: [medical advice based on the conversation]
            CATEGORY: [medical and care advice for family/medical staff/carers/allied health workers/doctors]
            
            If the conversation doesn't contain valuable medical insights, respond with: NO_CARD_NEEDED
            """
            
            try:
                new_card_response = client.generate_response(new_card_prompt, "", timeout=60)
                
                if "NO_CARD_NEEDED" not in new_card_response:
                    # Extract new cue card details
                    question_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:|$)', new_card_response, re.DOTALL | re.IGNORECASE)
                    answer_match = re.search(r'ANSWER:\s*(.+?)(?=CATEGORY:|$)', new_card_response, re.DOTALL | re.IGNORECASE)
                    category_match = re.search(r'CATEGORY:\s*(.+?)$', new_card_response, re.DOTALL | re.IGNORECASE)
                    
                    if question_match and answer_match:
                        new_question = question_match.group(1).strip()
                        new_answer = answer_match.group(1).strip()
                        category = category_match.group(1).strip() if category_match else "medical and care advice for family"
                        
                        # Create new cue card
                        new_card_id = vector_db.create_new_cue_card(
                            new_question,
                            new_answer,
                            category,
                            f"conversation_feedback_{session_id}",
                            model_used
                        )
                        
                        if new_card_id:
                            print(f"   ✅ Created new cue card: {new_question[:50]}...")
                            new_cards_created += 1
                        else:
                            print(f"   ❌ Failed to create new cue card")
                    else:
                        print(f"   ❌ Could not parse new card response")
                else:
                    print(f"   ➡️  No new insights found in conversation")
                    
            except Exception as e:
                print(f"   ❌ Error generating new card: {e}")
    
    print(f"\n🎉 Cue card update complete!")
    print(f"   📝 Updated {updates_made} existing cue cards")
    print(f"   🆕 Created {new_cards_created} new cue cards")
    
    return {
        "updates_made": updates_made,
        "new_cards_created": new_cards_created,
        "conversations_processed": len(recent_conversations)
    }


if __name__ == "__main__":
    # Example usage - uncomment what you want to test:
    
    # Process a document
    # process_document("/Users/alexander/Library/CloudStorage/Dropbox/Personal Research/cortex_bridge/paper/bsp_2.pdf")
    
    # Update cue cards from recent conversations
    update_cue_cards_from_conversations(days_back=1)