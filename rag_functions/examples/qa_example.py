#!/usr/bin/env python3
"""
Example of using the Q&A functionality with custom prompts
"""

from llm_analysis import analyze_with_custom_prompt
from config import RAGConfig

# Example Q&A templates matching your format
QA_TEMPLATES = {
    "detailed_qa": """<start_of_turn>system
You are a knowledgeable assistant. Answer questions thoroughly based on the provided context.
<end_of_turn>

<start_of_turn>user
Reference Material: {context}

Question: {prompt}

Please provide:
1. A direct answer to the question
2. Supporting evidence from the context
3. Any relevant additional insights
4. Confidence level in your answer
<end_of_turn>

<start_of_turn>model""",
    
    "concise_qa": """<start_of_turn>system
Provide brief, accurate answers based on the context.
<end_of_turn>

<start_of_turn>user
Context: {context}

Question: {prompt}

Answer concisely in 1-3 sentences.
<end_of_turn>

<start_of_turn>model""",
    
    "comparative_qa": """<start_of_turn>system
You are an analytical assistant that compares and contrasts information.
<end_of_turn>

<start_of_turn>user
Document: {context}

Task: {prompt}

Provide a comparative analysis highlighting:
- Key similarities
- Notable differences
- Overall assessment
<end_of_turn>

<start_of_turn>model"""
}

def run_qa_analysis(document_text, questions, template_name="detailed_qa"):
    """
    Run Q&A analysis on a document with multiple questions
    
    Args:
        document_text: The document content to analyze
        questions: List of questions to ask
        template_name: Which template to use from QA_TEMPLATES
    
    Returns:
        Dictionary of question->answer pairs
    """
    template = QA_TEMPLATES.get(template_name, QA_TEMPLATES["detailed_qa"])
    results = {}
    
    # Optional: use verbose mode to see progress
    config = RAGConfig(verbose=True)
    
    for question in questions:
        print(f"\nProcessing: {question}")
        response = analyze_with_custom_prompt(
            context=document_text,
            prompt=question,
            template=template,
            config=config
        )
        results[question] = response
    
    return results

# Example usage
if __name__ == "__main__":
    # Sample document text
    sample_context = """
    The company reported a 15% increase in revenue for Q3 2024, driven primarily by 
    strong performance in the cloud services division. Operating margins improved to 
    22% from 18% in the previous quarter. The CEO attributed this growth to strategic 
    investments in AI infrastructure and expanded partnerships with enterprise clients.
    
    Key highlights include:
    - Cloud revenue up 32% year-over-year
    - New customer acquisitions increased by 45%
    - R&D spending increased to 12% of revenue
    - Guidance for Q4 remains optimistic with expected 18-20% growth
    """
    
    # Questions to ask
    questions = [
        "What was the revenue growth in Q3 2024?",
        "What factors contributed to the improved performance?",
        "What are the future growth expectations?"
    ]
    
    # Run detailed Q&A
    print("=== DETAILED Q&A ===")
    detailed_results = run_qa_analysis(sample_context, questions, "detailed_qa")
    
    # Run concise Q&A
    print("\n\n=== CONCISE Q&A ===")
    concise_results = run_qa_analysis(sample_context, questions, "concise_qa")
    
    # Save results
    with open("qa_results.txt", "w") as f:
        f.write("DETAILED ANSWERS\n" + "="*50 + "\n\n")
        for q, a in detailed_results.items():
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
        
        f.write("\n\nCONCISE ANSWERS\n" + "="*50 + "\n\n")
        for q, a in concise_results.items():
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
    
    print("\nResults saved to qa_results.txt")