from rag_functions.ml.vector_operations import select_optimal_templates, analyze_document_type
from rag_functions.ml.cue_card_extraction import extract_cue_cards
from rag_functions.templates.prompt_templates import get_template

def process_medical_document(content: str, task: str = "", gemma_client=None):
    templates = select_optimal_templates(content, task)
    if not templates:
        return None
    
    template_name, confidence = templates[0]
    template = get_template(template_name)
    
    response = None
    cue_cards = []
    if gemma_client:
        response = gemma_client.generate_response(
            task or "Process this medical document",
            content[:5000],  # Limit content size
            prompt_template=template,
            timeout=120
        )
        cue_cards = extract_cue_cards(response, context_type="medical")
    
    return {
        'template_used': template_name,
        'confidence': confidence,
        'raw_response': response,
        'cue_cards': cue_cards,
        'document_type_scores': analyze_document_type(content)
    }