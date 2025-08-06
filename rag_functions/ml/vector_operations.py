from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder

def select_optimal_templates(content: str, task_description: str = "", similarity_threshold: float = 0.3, top_k: int = 3) -> List[Tuple[str, float]]:
    from rag_functions.templates.prompt_templates import ALL_TEMPLATES
    
    encoder = get_encoder()
    input_text = f"{content[:1000]} {task_description}".strip()
    content_vector = encoder.encode([input_text])[0]
    
    similarities = []
    for name, template in ALL_TEMPLATES.items():
        template_text = f"{template.description}. {', '.join(template.best_for)}"
        template_vector = encoder.encode([template_text])[0]
        sim = cosine_similarity([content_vector], [template_vector])[0][0]
        if sim >= similarity_threshold:
            similarities.append((name, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def analyze_document_type(content: str) -> Dict[str, float]:
    encoder = get_encoder()
    content_vector = encoder.encode([content[:1000]])[0]
    
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
    return get_encoder().encode(sentences)