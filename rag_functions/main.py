# main.py
# to use this
# KMP_DUPLICATE_LIB_OK=TRUE python main.py

from rag_functions.ocr_layout_copy import extract_text_and_layout
#from ocr_layout import extract_tables_pymupdf
from rag_functions.semantic_parser import parse_document
from rag_functions.tools import perform_calculations
from rag_functions.retrieval import setup_vector_db, retrieve_references
from rag_functions.llm_analysis import analyze_with_llm

import sys
import json

# Check dependencies
try:
    import layoutparser as lp
except ImportError:
    print("Warning: layoutparser not available")

def process_document(file_path, reference_texts, reference_meta=None, rag_config=None):
    """
    Process a document using Gemma models for analysis.
    
    Args:
        file_path: Path to the document to analyze
        reference_texts: List of reference document texts for context
        reference_meta: Optional metadata for references
        rag_config: Optional RAG configuration (str preset name or RAGConfig object)
    """
    # Import config
    from rag_functions.config import get_config, RAGConfig
    
    # Handle config parameter
    if rag_config is None:
        config = get_config("balanced")  # Default preset
    elif isinstance(rag_config, str):
        config = get_config(rag_config)  # Get preset by name
    elif isinstance(rag_config, RAGConfig):
        config = rag_config  # Use provided config
    else:
        config = get_config("balanced")  # Fallback to default
    
    # Process document
    if config.verbose:
        print(f"Processing: {file_path} ({rag_config if isinstance(rag_config, str) else 'custom'})")
    
    txt = extract_text_and_layout(file_path)
    #tables = extract_tables_pymupdf(file_path)
    
    if config.save_intermediate_results:
        with open('document.txt', 'w') as file:
            file.write(txt)

    parsed = parse_document(txt)

    if config.save_intermediate_results:
        with open('parsed.txt', 'w') as file:
            file.write(parsed)
    
    # # Save extracted tables
    # with open('tables.json', 'w') as file:
    #     json.dump(tables, file, indent=2)

    calc = perform_calculations(parsed) if config.include_calculations else None

    # Build retrieval store from reference document corpus
    references = []
    if config.include_references and reference_texts:
        vectorstore = setup_vector_db(reference_texts, reference_meta)
        references = retrieve_references(vectorstore, parsed, k=config.max_reference_chunks)

    report = analyze_with_llm(parsed, calc, references, config)
    return report

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process documents using Gemma RAG")
    parser.add_argument("--file", default="Improved Specificity Clusters.pdf", 
                       help="Path to document file")
    parser.add_argument("--config", default="balanced", 
                       choices=["fast", "quality", "balanced", "debug"],
                       help="Configuration preset to use")
    parser.add_argument("--model", help="Override default Gemma model")
    parser.add_argument("--template", help="Override default prompt template")
    parser.add_argument("--no-template", action="store_true",
                       help="Disable prompt templates (use basic prompts)")
    parser.add_argument("--no-references", action="store_true", 
                       help="Skip reference document analysis")
    parser.add_argument("--no-calculations", action="store_true",
                       help="Skip calculations")
    args = parser.parse_args()
    
    # Load your corpus of reference documents as raw text
    references = ["reference doc 1...", "reference doc 2...", "..."] if not args.no_references else []
    meta = [{"doc_id": "DocA"}, {"doc_id": "DocB"}, {}] if not args.no_references else None
    
    # Override config if specific options provided
    if args.model or args.template or args.no_template or args.no_references or args.no_calculations:
        from rag_functions.config import get_config
        config = get_config(args.config)
        if args.model:
            config.default_model = args.model
        if args.template:
            config.default_template = args.template
        if args.no_template:
            config.use_prompt_template = False
        if args.no_references:
            config.include_references = False
        if args.no_calculations:
            config.include_calculations = False
        output = process_document(args.file, references, meta, config)
    else:
        # Use preset directly
        output = process_document(args.file, references, meta, args.config)

    # Save output
    output_file = f'report_{args.config}.txt'
    with open(output_file, 'w') as file:
        file.write(output)

    print(f"\nReport saved to: {output_file}")
    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    print(output)

