# llm_analysis.py

import sys
from pathlib import Path
from typing import Optional

# Add program_files to path to import Gemma client
sys.path.append(str(Path(__file__).parent.parent / "program_files"))

# Import Gemma client - using optimized version for better performance
from ai.optimized_gemma_client import OptimizedGemmaClient
from ai.gemma_client import GemmaClient

# Import RAG configuration and prompt templates
from .config import RAGConfig, get_config
from .prompt_templates import get_template

def analyze_with_llm(parsed_entities, calc_results, reference_chunks, config: Optional[RAGConfig] = None):
    """
    Construct a RAG‑style prompt combining parsed text-chunk entities,
    calculations, and retrieved reference chunks to generate a final report.
    
    Args:
        parsed_entities: Extracted entities from document
        calc_results: Calculation results
        reference_chunks: Retrieved similar reference document chunks
        config: RAG configuration (uses default if None)
    """
    # Use default config if none provided
    if config is None:
        config = get_config("balanced")
    
    # Initialize Gemma client based on configuration
    if config.use_optimized_client:
        # OptimizedGemmaClient automatically selects best model based on prompt
        client = OptimizedGemmaClient()
        if config.verbose:
            print(f"🤖 Using OptimizedGemmaClient with auto model selection")
    else:
        # Basic client with configured model
        client = GemmaClient(model=config.default_model, base_url=config.ollama_base_url)
        if config.verbose:
            print(f"🤖 Using GemmaClient with model: {config.default_model}")
    
    # Prepare context and prompt
    context_parts = []
    
    # Add parsed entities
    context_parts.append(f"Document Analysis:\n{parsed_entities}")
    
    # Add calculations if available
    if config.include_calculations and calc_results:
        context_parts.append(f"Calculations:\n{calc_results}")
    
    # Add reference context if available
    reference_context = "\n\n".join(reference_chunks[:config.max_reference_chunks])
    if config.include_references and reference_context:
        context_parts.append(f"Reference Documents:\n{reference_context}")
    
    context = "\n\n".join(context_parts)
    
    # Determine the prompt template to use
    prompt_template = None
    if config.use_prompt_template:
        if config.custom_template:
            prompt_template = config.custom_template
        else:
            prompt_template = get_template(config.default_template)
            if not prompt_template:
                if config.verbose:
                    print(f"⚠️ Template '{config.default_template}' not found, falling back to basic prompt")
    
    # Create the main prompt
    if config.detailed_report:
        prompt = "Provide a comprehensive analysis of this document including key findings, analysis of entities, interpretation of data, and actionable recommendations."
    else:
        prompt = "Provide a concise analysis of this document highlighting the main points and insights."
    
    if config.verbose:
        print(f"📝 Generated prompt length: {len(prompt)} characters")
        if prompt_template:
            print(f"📋 Using template: {config.custom_template or config.default_template}")
    
    # Generate response using Gemma
    try:
        if config.use_optimized_client:
            # Use optimized method that includes model selection and latency monitoring
            response = client.generate_response_optimized(
                prompt=prompt,
                context=context,
                prompt_template=prompt_template,
                vector_context={
                    "reference_count": len(reference_chunks),
                    "has_calculations": bool(calc_results),
                    "prefer_fast": config.prefer_fast_models
                }
            )
        else:
            # Use basic generation method
            response = client.generate_response(
                prompt=prompt,
                context=context,
                prompt_template=prompt_template,
                timeout=config.request_timeout
            )
    except Exception as e:
        if config.verbose:
            print(f"❌ Error generating response: {e}")
        response = None
    
    # Check if response was generated successfully
    if response is None:
        print("⚠️ Failed to generate response from Gemma. Using fallback response.")
        return "Analysis could not be completed. Please check if Ollama server is running."
    
    return response
