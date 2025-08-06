from typing import List
from dataclasses import dataclass
import re
from sklearn.cluster import KMeans
from .vector_operations import vectorize_sentences
import sys, os

@dataclass
class CueCard:
    title: str
    context: str
    key_points: List[str]
    priority: str = "medium"
    category: str = "medical"

def extract_cue_cards(llm_output: str, context_type: str = "medical") -> List[CueCard]:
    if not llm_output or len(llm_output.strip()) < 50:
        return []
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', llm_output) if len(s.strip()) > 20]
    if len(sentences) < 2:
        return _simple_extract(llm_output, context_type)
    
    # Cluster sentences
    vectors = vectorize_sentences(sentences)
    n_clusters = min(max(2, len(sentences) // 3), 5)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    clusters = {}
    for sentence, label in zip(sentences, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentence)
    
    # Generate cue cards with Gemma
    cue_cards = []
    sys.path.append(str(os.path.join(os.path.dirname(__file__), '..', '..', 'program_files')))
    from ai.gemma_client import GemmaClient
    
    gemma_client = GemmaClient(model="gemma3n:e4b")
    
    for cluster_sentences in clusters.values():
        if cluster_sentences:
            cluster_text = ' '.join(cluster_sentences)
            prompt = f"""Create a medical cue card from this content.
Format: [Medical Situation]: [Actionable Care Advice]

Content: {cluster_text}

Create ONE concise line describing the situation and the care advice."""
            
            response = gemma_client.generate_response("Create medical cue card", prompt, timeout=60)
            
            if response and ':' in response:
                lines = [line.strip() for line in response.split('\n') if ':' in line and len(line.strip()) > 10]
                if lines:
                    parts = lines[0].split(':', 1)
                    if len(parts) == 2:
                        context_part = parts[0].strip().replace('*', '').replace('#', '')
                        advice_part = parts[1].strip().replace('*', '').replace('#', '')
                        cue_cards.append(CueCard(
                            title=context_part,
                            context=context_part,
                            key_points=[advice_part]
                        ))
    
    return cue_cards if cue_cards else _simple_extract(llm_output, context_type)

def _simple_extract(content: str, context_type: str) -> List[CueCard]:
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30][:3]
    cue_cards = []
    for i, para in enumerate(paragraphs):
        sentences = [s.strip() for s in re.split(r'[.!?]+', para) if len(s.strip()) > 15][:4]
        if sentences:
            cue_cards.append(CueCard(
                title=f"{context_type} {i+1}",
                context=context_type,
                key_points=sentences[1:] if len(sentences) > 1 else sentences
            ))
    return cue_cards

def format_cue_cards(cue_cards: List[CueCard]) -> str:
    return '\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in cue_cards])