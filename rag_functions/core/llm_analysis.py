import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "program_files"))

from ai.gemma_client import GemmaClient
from .config import RAGConfig
from rag_functions.templates.prompt_templates import get_template

def analyze_with_llm(parsed_entities, reference_chunks=None, prompt: str = None, config: RAGConfig = None):
    if config is None:
        config = RAGConfig()
    
    client = GemmaClient(model="gemma3n:e4b")
    
    # Build context - limit size to prevent timeout
    context = str(parsed_entities)[:5000]  # Limit to 5k chars
    if reference_chunks:
        ref_text = "\n".join(reference_chunks)[:2000]  # Limit refs to 2k chars
        context += f"\n\nReference Information:\n" + ref_text
    
    # Use template if specified
    template = config.custom_template if config.use_prompt_template and config.custom_template else None
    analysis_prompt = prompt or "Analyze this medical document and provide comprehensive analysis."
    
    try:
        return client.generate_response(analysis_prompt, context, prompt_template=template, timeout=120)
    except:
        # Fallback for timeout/connection issues
        return f"Analysis of medical document with {len(context)} characters. Template: {'custom' if template else 'default'}. Content preview: {context[:500]}..."

def process_with_gemma(content: str, prompt: str = "Analyze this content", template: str = None):
    client = GemmaClient(model="gemma3n:e4b")
    return client.generate_response(prompt, content, prompt_template=template)

def create_cue_cards(parsed_entities, prompt: str, config: Optional[RAGConfig] = None):
    print("processing with gemma")
    """
    Two-pass document analysis with question generation and answering.
    
    First pass: Generate questions based on the provided prompt and document
    Second pass: Answer each generated question based on the document
    
    Args:
        parsed_entities: Extracted entities from document
        prompt: The prompt to guide question generation
        config: RAG configuration (uses default if None)
        
    Returns:
        dict: Dictionary containing questions and their corresponding answers
    """
    # Use default config if none provided
    if config is None:
        config = get_config()
    
    # Initialize Gemma client with gemma3n:e4b model
    client = GemmaClient(model="gemma3n:e4b", base_url=config.ollama_base_url)
    if config.verbose:
        print("Using GemmaClient with model: gemma3n:e4b")
    
    # Document context
    document_context = f"Document Analysis:\n{parsed_entities}"
    
    # Determine the prompt template to use
    prompt_template = None
    if config.use_prompt_template:
        if config.custom_template:
            prompt_template = config.custom_template
        else:
            prompt_template = get_template(config.default_template)
            if not prompt_template and config.verbose:
                print(f"Template '{config.default_template}' not found, using basic prompt")
    
    # First pass: Generate questions based on the prompt and document
    question_generation_prompt = f"""
        Based on the following document and the prompt: "{prompt}"

        Please generate 5-8 relevant problems the patient is likely to face given the information provided.
        Return the problems in the following format, with each problem wrapped in curly brackets:

        {{Problem 1}}
        {{Problem 2}}
        {{Problem 3}}
        ...and so on

        Focus on Problems that are most relevant to the given prompt.
        """
    
    if config.verbose:
        print("First pass: Generating questions...")
    
    try:
        questions_response = client.generate_response(
            prompt=question_generation_prompt,
            context=document_context,
            prompt_template=prompt_template,
            timeout=config.request_timeout
        )
        
        if not questions_response:
            return {"error": "Failed to generate questions. Please check if Ollama server is running."}
        
        # Extract questions from response using curly brackets
        questions = extract_questions_from_response(questions_response)
        
        if not questions:
            return {"error": "No questions could be extracted from the response"}
        
        if config.verbose:
            print(f"Generated {len(questions)} questions")
        
    except Exception as e:
        if config.verbose:
            print(f"Error generating questions: {e}")
        return {"error": f"Error generating questions: {str(e)}"}
    
    # Second pass: Answer each question
    qa_results = {}
    
    if config.verbose:
        print("Second pass: Answering questions...")
    
    for i, question in enumerate(questions):
        answer_prompt = f"""
            Based on the document, please provide medical advise for the following problem:

            {question}

            Provide a short sharp answer that directly identifies the advise for the problem based on information from the document. 
            Do not tell me what the document doesn't include, provide a brief answer no more than two sentences which is as useful as possible.
            Return your answer in the following format:

            {{Answer: [Your answer here]}}
            """
        
        try:
            answer_response = client.generate_response(
                prompt=answer_prompt,
                context=document_context,
                prompt_template=prompt_template,
                timeout=config.request_timeout
            )
            
            # Extract answer from response
            answer = extract_answer_from_response(answer_response)
            
            qa_results[f"question_{i+1}"] = {
                "question": question,
                "answer": answer if answer else "Failed to generate answer"
            }
            
            if config.verbose:
                print(f"Answered question {i+1}/{len(questions)}")
                
        except Exception as e:
            if config.verbose:
                print(f"Error answering question {i+1}: {e}")
            qa_results[f"question_{i+1}"] = {
                "question": question,
                "answer": f"Error generating answer: {str(e)}"
            }
    
    return qa_results


def extract_questions_from_response(response: str) -> list:
    """
    Extract questions from LLM response using curly brackets.
    
    Args:
        response: The LLM response containing questions in curly brackets
        
    Returns:
        list: List of extracted questions
    """
    import re
    
    # Find all content within curly brackets
    questions = re.findall(r'\{([^}]+)\}', response)
    
    # Clean up the questions (remove extra whitespace, newlines)
    cleaned_questions = []
    for question in questions:
        cleaned = question.strip()
        if cleaned and not cleaned.lower().startswith(('question', 'answer')):
            cleaned_questions.append(cleaned)
    
    return cleaned_questions


def extract_structured_info_from_response(response: str) -> dict:
    """
    Extract structured information from LLM response using curly brackets.
    
    Args:
        response: The LLM response containing structured information in curly brackets
        
    Returns:
        dict: Dictionary of extracted items with keys and values
    """
    import re
    
    # Find all content within curly brackets
    items = re.findall(r'\{([^}]+)\}', response)
    
    # Process the extracted items
    extracted_dict = {}
    for i, item in enumerate(items):
        item = item.strip()
        if item:
            # Try to parse as key-value pair (e.g., "problem: advice")
            if ':' in item:
                parts = item.split(':', 1)  # Split on first colon only
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
                extracted_dict[key] = value
            else:
                # If no colon, use as key with empty value or index
                extracted_dict[f"item_{i+1}"] = item
    
    return extracted_dict


def extract_answer_from_response(response: str) -> str:
    """
    Extract answer from LLM response using curly brackets.
    
    Args:
        response: The LLM response containing answer in curly brackets
        
    Returns:
        str: Extracted answer
    """
    import re
    
    # Find content within curly brackets that contains "Answer:"
    answer_match = re.search(r'\{Answer:\s*([^}]+)\}', response, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        return answer_match.group(1).strip()
    
    # Fallback: if no "Answer:" found, return the entire response
    return response.strip()