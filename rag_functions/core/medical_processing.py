"""
Medical document processing workflows
"""
from typing import Dict, Optional
from rag_functions.ml.vector_operations import select_optimal_templates, analyze_document_type
from rag_functions.ml.cue_card_extraction import extract_cue_cards, format_cue_cards
from rag_functions.templates.prompt_templates import get_template

def process_medical_document(content: str, task: str = "", gemma_client=None):
    """
    Complete workflow: analyze -> select template -> generate -> extract cue cards
    
    Args:
        content: Medical document content to process
        task: Optional task description
        gemma_client: GemmaClient instance (optional, for standalone use)
    
    Returns:
        Dictionary with template info, analysis results, and cue cards
    """
    # 1. Analyze document and select templates
    templates = select_optimal_templates(content, task)
    if not templates:
        print("No suitable templates found")
        return None
    
    # 2. Use best template
    template_name, confidence = templates[0]
    template = get_template(template_name)
    
    print(f"Using template: {template_name} (confidence: {confidence:.2f})")
    
    # 3. Generate response (requires external client)
    response = None
    if gemma_client:
        response = gemma_client.generate_response(
            prompt=task or "Process this medical document",
            context=content,
            prompt_template=template
        )
        
        # 4. Extract and format cue cards
        cue_cards = extract_cue_cards(response, context_type="medical")
        formatted_output = format_cue_cards(cue_cards)
    else:
        cue_cards = []
        formatted_output = "No client provided - cannot generate analysis"
    
    return {
        'template_used': template_name,
        'confidence': confidence,
        'raw_response': response,
        'cue_cards': cue_cards,
        'formatted_cards': formatted_output,
        'document_type_scores': analyze_document_type(content)
    }