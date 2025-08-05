"""
Vector operations for document analysis and template selection
"""
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global encoder for efficiency
_encoder = None

def get_encoder():
    """Get or initialize the sentence transformer encoder"""
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder

def select_optimal_templates(content: str, task_description: str = "", 
                           similarity_threshold: float = 0.3, top_k: int = 3,
                           templates_dict: Dict = None) -> List[Tuple[str, float]]:
    """Select templates using vector similarity"""
    if not templates_dict:
        from rag_functions.templates.prompt_templates import ALL_TEMPLATES
        templates_dict = ALL_TEMPLATES
        
    encoder = get_encoder()
    
    # Vectorize input
    input_text = f"{content[:1000]} {task_description}".strip()
    content_vector = encoder.encode([input_text])[0]
    
    # Calculate similarities with templates
    similarities = []
    for name, template in templates_dict.items():
        template_text = f"{template.description}. {', '.join(template.best_for)}"
        template_vector = encoder.encode([template_text])[0]
        sim = cosine_similarity([content_vector], [template_vector])[0][0]
        if sim >= similarity_threshold:
            similarities.append((name, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def analyze_document_type(content: str) -> Dict[str, float]:
    """Analyze document type using vector similarity"""
    encoder = get_encoder()
    content_vector = encoder.encode([content[:1000]])[0]
    
    # Category descriptions for comparison
    categories = {
        'intake': "Patient intake, medical history, chief complaint, new patient visit",
        'clinical': "Clinical notes, SOAP notes, progress notes, patient encounters", 
        'report': "Medical reports, diagnostic reports, pathology results, radiology findings",
        'diagnostic': "Differential diagnosis, clinical analysis, diagnostic reasoning",
        'treatment': "Treatment planning, medication management, care coordination"
    }
    
    scores = {}
    for category, description in categories.items():
        cat_vector = encoder.encode([description])[0]
        scores[category] = cosine_similarity([content_vector], [cat_vector])[0][0]
    
    return scores

def vectorize_sentences(sentences: List[str]) -> List:
    """Vectorize a list of sentences"""
    encoder = get_encoder()
    return encoder.encode(sentences)