# llm_analysis.py

import sys
from pathlib import Path
from typing import Optional

# Add program_files to path to import Gemma client
sys.path.append(str(Path(__file__).parent.parent / "program_files"))

# Import Gemma client - using optimized version for better performance
from ai.optimized_gemma_client import OptimizedGemmaClient
from ai.gemma_client import GemmaClient

# Import RAG configuration
from .config import RAGConfig, get_config

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
    context = "\n\n".join(reference_chunks[:config.max_reference_chunks])
    
    # Build prompt based on configuration
    prompt_parts = ["You are an AI assistant analyzing documents."]
    
    prompt_parts.append(f"\nParsed document entities:\n{parsed_entities}")
    
    if config.include_calculations and calc_results:
        prompt_parts.append(f"\nCalculation results:\n{calc_results}")
    
    if config.include_references and context:
        prompt_parts.append(f"\nReference context (from similar documents):\n{context}")
    
    if config.detailed_report:
        prompt_parts.append("""
Please generate a structured analysis report for this document based on the above.
Include:
1. Summary of key findings
2. Analysis of the parsed entities
3. Interpretation of calculations (if any)
4. Relevant references and their implications
5. Conclusions and recommendations
""")
    else:
        prompt_parts.append("\nPlease provide a concise analysis of this document.")
    
    prompt = "\n".join(prompt_parts)
    
    if config.verbose:
        print(f"📝 Generated prompt length: {len(prompt)} characters")
    
    # Generate response using Gemma
    try:
        if config.use_optimized_client:
            # Use optimized method that includes model selection and latency monitoring
            response = client.generate_response_optimized(
                prompt=prompt,
                context="Document analysis with RAG",
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
                context="Document analysis",
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
