"""
Cue card extraction from medical documents
"""
from typing import List
from dataclasses import dataclass
import re
from sklearn.cluster import KMeans
from .vector_operations import vectorize_sentences

@dataclass
class CueCard:
    """Container for a medical cue card with context and key information"""
    title: str
    context: str  # medical, situational, clinical, etc.
    key_points: List[str]
    priority: str  # high, medium, low
    category: str  # diagnosis, treatment, monitoring, etc.

def extract_cue_cards(llm_output: str, context_type: str = "medical") -> List[CueCard]:
    """Extract cue cards using sentence clustering and Gemma generation"""
    if not llm_output or len(llm_output.strip()) < 50:
        return []
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', llm_output) if len(s.strip()) > 20]
    
    if len(sentences) < 2:
        return _simple_extract(llm_output, context_type)
    
    # Vectorize and cluster sentences
    vectors = vectorize_sentences(sentences)
    n_clusters = min(max(2, len(sentences) // 3), 5)  # 2-5 clusters
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
    except:
        return _simple_extract(llm_output, context_type)
    
    # Group sentences by cluster
    clusters = {}
    for sentence, label in zip(sentences, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentence)
    
    # Use Gemma to generate cue cards for each cluster
    cue_cards = []
    try:
        import sys, os
        sys.path.append(str(os.path.join(os.path.dirname(__file__), '..', '..', 'program_files')))
        from ai.gemma_client import GemmaClient
        
        gemma_client = GemmaClient(model="gemma3n:e4b")
        
        for cluster_sentences in clusters.values():
            if len(cluster_sentences) > 0:
                cluster_text = ' '.join(cluster_sentences)
                
                # Ask Gemma to create cue card in the specified format  
                prompt = f"""Create a medical cue card from this content.
Format: [Medical Situation]: [Actionable Care Advice]

Content: {cluster_text}

Example: "Acute MI with ST elevation: Give aspirin 325mg, prepare for emergency catheterization"

Create ONE concise line describing the situation and the care advice."""
                
                response = gemma_client.generate_response(
                    prompt="Create medical cue card",
                    context=prompt,
                    timeout=15
                )
                
                if response and ':' in response:
                    # Clean up response and find the best line with ':'
                    lines = [line.strip() for line in response.split('\n') if ':' in line and len(line.strip()) > 10]
                    
                    if lines:
                        # Use the first good line with context:advice format
                        best_line = lines[0]
                        parts = best_line.split(':', 1)
                        
                        if len(parts) == 2:
                            context_part = parts[0].strip().replace('*', '').replace('#', '')
                            advice_part = parts[1].strip().replace('*', '').replace('#', '')
                                                        
                            cue_cards.append(CueCard(
                                title=context_part,
                                context=context_part,
                                key_points=[advice_part],
                                priority="medium",
                                category="medical"
                            ))
    except Exception as e:
        # Fallback if Gemma fails
        return _simple_extract(llm_output, context_type)
    
    return cue_cards if cue_cards else _simple_extract(llm_output, context_type)

def _simple_extract(content: str, context_type: str) -> List[CueCard]:
    """Simple fallback extraction"""
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30][:3]
    
    cue_cards = []
    for i, para in enumerate(paragraphs):
        sentences = [s.strip() for s in re.split(r'[.!?]+', para) if len(s.strip()) > 15][:4]
        if sentences:
            cue_cards.append(CueCard(
                title=f"{context_type} {i+1}",
                context=context_type,
                key_points=sentences[1:] if len(sentences) > 1 else sentences,
                priority="medium",
                category=context_type
            ))
    
    return cue_cards

def format_cue_cards(cue_cards: List[CueCard]) -> str:
    """Format cue cards into readable text"""
    if not cue_cards:
        return "No cue cards generated."
    
    return '\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in cue_cards])