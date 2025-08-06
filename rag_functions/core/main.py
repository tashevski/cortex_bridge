from rag_functions.utils.ocr_layout_copy import extract_text_and_layout
from rag_functions.utils.semantic_parser import parse_document
from rag_functions.utils.retrieval import setup_vector_db, retrieve_references
from .llm_analysis import analyze_with_llm
from rag_functions.templates.prompt_templates import get_template
from rag_functions.ml.vector_operations import select_optimal_templates, analyze_document_type
from rag_functions.ml.cue_card_extraction import extract_cue_cards
from .config import get_config

def process_document(file_path, reference_texts=None, use_medical_templates=True, generate_cue_cards=True, context_type="medical"):
    config = get_config()
    
    # Extract and parse
    txt = extract_text_and_layout(file_path)
    parsed = parse_document(txt)
    
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
    
    # Generate cue cards
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