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
import re
import json


def process_document(file_path, reference_texts=None, use_medical_templates=True, generate_cue_cards=True, context_type="medical"):
    config = get_config()
    
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
    print(adaptive_prompts)
    
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
        #print(contextual_responses[prompt])


    # Setup references
    references = []
    if reference_texts:
        vectorstore = setup_vector_db(reference_texts, None)
        references = retrieve_references(vectorstore, parsed, k=config.max_reference_chunks)
    
    # Medical template selection
    if use_medical_templates:
        templates = select_optimal_templates(parsed, "Analyze medical document")
        if templates:
            template = get_template(templates[0][0])
            config.custom_template = template
            config.use_prompt_template = True
    
    # Analyze
    results = analyze_with_llm(parsed, references, config=config)
    
    # Generate context response advice 
    cue_cards = {}
    if generate_cue_cards:
        if isinstance(results, dict):
            for key, content in results.items():
                cue_cards[key] = {'cue_cards': extract_cue_cards(content, context_type)}
        else:
            cue_cards['main'] = {'cue_cards': extract_cue_cards(results, context_type)}
    
    return {
        'analysis': results,
        'cue_cards': cue_cards,
        'template_info': templates[0] if use_medical_templates and templates else None,
        'document_type_scores': analyze_document_type(parsed) if use_medical_templates else None
    }


if __name__ == "__main__":
    process_document("/Users/alexander/Library/CloudStorage/Dropbox/Personal Research/cortex_bridge/paper/bsp_2.pdf")