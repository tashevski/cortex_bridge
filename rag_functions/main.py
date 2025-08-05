# main.py
# to use this
# KMP_DUPLICATE_LIB_OK=TRUE python main.py

from rag_functions.ocr_layout_copy import extract_text_and_layout
#from ocr_layout import extract_tables_pymupdf
from rag_functions.semantic_parser import parse_document
from rag_functions.retrieval import setup_vector_db, retrieve_references
from rag_functions.llm_analysis import analyze_with_llm, process_with_gemma

import sys
import json

# Check dependencies
try:
    import layoutparser as lp
except ImportError:
    print("Warning: layoutparser not available")

def process_document(file_path, reference_texts=None, reference_meta=None, rag_config=None, prompt=None):
    """
    Process a document using Gemma models for analysis.
    
    Args:
        file_path: Path to the document to analyze
        reference_texts: List of reference document texts for context (optional)
        reference_meta: Optional metadata for references
        rag_config: Optional RAG configuration
        prompt: Optional prompt for structured information extraction
    """
    # Import config
    from rag_functions.config import get_config, RAGConfig
    
    # Use provided config or default
    if rag_config is None:
        config = get_config()
    else:
        config = rag_config
    
    # Process document
    if config.verbose:
        print(f"Processing: {file_path}")
    
    txt = extract_text_and_layout(file_path)
    #tables = extract_tables_pymupdf(file_path)
    
    if config.save_intermediate_results:
        with open('document.txt', 'w') as file:
            file.write(txt)

    parsed = parse_document(txt)

    if config.save_intermediate_results:
        with open('parsed.txt', 'w') as file:
            file.write(parsed)
    

    # Build retrieval store from reference document corpus
    references = []
    if config.include_references and reference_texts:
        vectorstore = setup_vector_db(reference_texts, reference_meta)
        references = retrieve_references(vectorstore, parsed, k=config.max_reference_chunks)

    # Analyze with LLM - returns dict if references provided, string otherwise
    if prompt:
        # Use prompt-based analysis
        results = analyze_with_llm(parsed, prompt=prompt, config=rag_config)
    else:
        # Use reference-based analysis
        results = analyze_with_llm(parsed, references, config=rag_config)
    
    #results = process_with_gemma(parsed, reference_texts)
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process documents using Gemma RAG")
    parser.add_argument("--file", default="Improved Specificity Clusters.pdf", 
                       help="Path to document file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--template", help="Override default prompt template")
    parser.add_argument("--no-template", action="store_true",
                       help="Disable prompt templates (use basic prompts)")
    parser.add_argument("--no-references", action="store_true", 
                       help="Skip reference document analysis")
    args = parser.parse_args()
    
    # Load your corpus of reference documents as raw text
    references = ["reference doc 1...", "reference doc 2...", "..."] if not args.no_references else []
    meta = [{"doc_id": "DocA"}, {"doc_id": "DocB"}, {}] if not args.no_references else None
    
    # Create config if options provided
    if args.verbose or args.template or args.no_template or args.no_references:
        from rag_functions.config import RAGConfig
        config = RAGConfig()
        if args.verbose:
            config.verbose = True
        if args.template:
            config.default_template = args.template
        if args.no_template:
            config.use_prompt_template = False
        if args.no_references:
            config.include_references = False
        output = process_document(args.file, references, meta, config)
    else:
        # Use default config
        output = process_document(args.file, references, meta)

    # Save output - handle both string and dictionary responses
    if isinstance(output, dict):
        # Save each reference analysis separately
        for key, content in output.items():
            output_file = f'report_{key}.txt'
            with open(output_file, 'w') as file:
                file.write(content)
            print(f"Report saved to: {output_file}")
        
        # Also save combined report
        combined_file = 'report_combined.txt'
        with open(combined_file, 'w') as file:
            for key, content in output.items():
                file.write(f"\n{'='*50}\n")
                file.write(f"{key.upper()}\n")
                file.write(f"{'='*50}\n\n")
                file.write(content)
                file.write("\n\n")
        print(f"\nCombined report saved to: {combined_file}")
        
        # Display summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Generated {len(output)} reference comparisons")
    else:
        # Single report output
        output_file = 'report.txt'
        with open(output_file, 'w') as file:
            file.write(output)

        print(f"\nReport saved to: {output_file}")
        print("\n" + "="*50)
        print("ANALYSIS REPORT")
        print("="*50)
        print(output)

