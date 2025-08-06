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


def store_cue_cards_in_db(vector_db, document_path, cue_cards, prompt_type):
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
                "session_id": f"rag_session_{timestamp.replace(':', '-')}"
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
        store_cue_cards_in_db(vector_db, file_path, contextual_responses[prompt], prompt)
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


if __name__ == "__main__":
    process_document("/Users/alexander/Library/CloudStorage/Dropbox/Personal Research/cortex_bridge/paper/bsp_2.pdf")