# llm_analysis.py

import sys
from pathlib import Path
from typing import Optional

# Add program_files to path to import Gemma client
sys.path.append(str(Path(__file__).parent.parent.parent / "program_files"))

# Import Gemma client
from ai.gemma_client import GemmaClient

# Import RAG configuration and prompt templates
from .config import RAGConfig, get_config
from rag_functions.templates.prompt_templates import get_template

def analyze_with_llm(parsed_entities, reference_chunks=None, prompt: str = None, config: Optional[RAGConfig] = None):
    """
    Analyze document with optional reference chunks or prompt-based structured extraction.
    
    If prompt provided: processes the prompt and extracts structured information from response
    If reference_chunks provided: generates individual response for each chunk
    If neither provided: generates single detailed report
    
    Args:
        parsed_entities: Extracted entities from document
        reference_chunks: Optional list of reference document chunks
        prompt: Optional prompt to process and extract structured information from
        config: RAG configuration (uses default if None)
        
    Returns:
        str: Single report if no reference chunks or prompt
        dict: Dictionary of responses keyed by chunk index if reference chunks provided
        dict: Dictionary of extracted structured information if prompt provided
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
    
    # Case 1: Prompt provided - process single prompt and extract structured information
    if prompt:
        if config.verbose:
            print(f"Processing with prompt: {prompt}")
        
        try:
            response = client.generate_response(
                prompt=prompt,
                context=document_context,
                prompt_template=prompt_template,
                timeout=config.request_timeout
            )
            
            if not response:
                return {"error": "Failed to generate response. Please check if Ollama server is running."}
            
            # Extract structured information from response using curly brackets
            #extracted_items = extract_structured_info_from_response(response)
            
            # if not extracted_items:
            #     return {"error": "No structured information could be extracted from the response", "raw_response": response}
            
            # if config.verbose:
            #     print(f"Extracted {len(extracted_items)} items from response")
            
            return response
            
        except Exception as e:
            if config.verbose:
                print(f"Error processing prompt: {e}")
            return {"error": f"Error processing prompt: {str(e)}"}
    
    # Case 2: No prompt but reference chunks provided - generate response for each
    elif reference_chunks and config.include_references:
        responses = {}
        chunks_to_process = reference_chunks[:config.max_reference_chunks]
        
        if config.verbose:
            print(f"Processing {len(chunks_to_process)} reference chunks individually")
        
        for i, chunk in enumerate(chunks_to_process):
            # Combine document context with this specific reference
            combined_context = f"{document_context}"
            
            # Prompt for comparing with reference
            prompt = "Based on the document context, please answer the question: {prompt}."
            
            try:
                response = client.generate_response(
                    prompt=prompt,
                    context=combined_context,
                    prompt_template=prompt_template,
                    timeout=config.request_timeout
                )
                responses[f"reference_{i}"] = response if response else "Failed to analyze this reference"
                
                if config.verbose:
                    print(f"Processed reference chunk {i+1}/{len(chunks_to_process)}")
                    
            except Exception as e:
                if config.verbose:
                    print(f"Error processing reference {i}: {e}")
                responses[f"reference_{i}"] = f"Error analyzing reference: {str(e)}"
        
        return responses
    
    # Case 3: No prompt and no reference chunks - generate single detailed report
    else:
        if config.detailed_report:
            prompt = "Provide a comprehensive analysis of this document including key findings, analysis of entities, interpretation of data, and actionable recommendations."
        else:
            prompt = "Provide a concise analysis of this document highlighting the main points and insights."
        
        if config.verbose:
            print(f"Generating single report. Prompt length: {len(prompt)} characters")
        
        try:
            response = client.generate_response(
                prompt=prompt,
                context=document_context,
                prompt_template=prompt_template,
                timeout=config.request_timeout
            )
            return response if response else "Analysis could not be completed. Please check if Ollama server is running."
        except Exception as e:
            if config.verbose:
                print(f"Error generating response: {e}")
            return "Analysis could not be completed. Please check if Ollama server is running."


def process_with_gemma(parsed_entities, prompt: str, config: Optional[RAGConfig] = None):
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

        Please identify all the relevant problems the patient is likely to face given the information provided, think about the different contexts the patient may be in. I expect there will be many problems in many different contexts, so please identify all of them.
        Return the problems in the following format, with each problem wrapped in curly brackets:

        {{Problem 1}}
        {{Problem 2}}
        {{Problem 3}}
        ...and so on

        Focus on Problems that are most relevant to the given prompt. Do not miss any problems. 
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
            Based on the document, please provide medical or care advice for the following problem:

            {question}

            Provide a short sharp answer that directly identifies the medical or care advise for the problem based on information from the document. 
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