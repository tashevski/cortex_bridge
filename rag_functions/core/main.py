# main.py
# to use this
# KMP_DUPLICATE_LIB_OK=TRUE python main.py

from rag_functions.utils.ocr_layout_copy import extract_text_and_layout
#from ocr_layout import extract_tables_pymupdf
from rag_functions.utils.semantic_parser import parse_document
from rag_functions.utils.retrieval import setup_vector_db, retrieve_references
from .llm_analysis import analyze_with_llm, process_with_gemma
from rag_functions.templates.prompt_templates import get_template
from rag_functions.ml.vector_operations import select_optimal_templates, analyze_document_type
from rag_functions.ml.cue_card_extraction import extract_cue_cards

import sys
import json
from copy import deepcopy

# Check dependencies
try:
    import layoutparser as lp
except ImportError:
    print("Warning: layoutparser not available")

def process_document(file_path, reference_texts=None, reference_meta=None, rag_config=None, prompt=None, 
                    use_medical_templates=True, generate_cue_cards=True, context_type="medical"):
    """
    Process a document using Gemma models for analysis with medical template selection and cue card generation.
    
    Args:
        file_path: Path to the document to analyze
        reference_texts: List of reference document texts for context (optional)
        reference_meta: Optional metadata for references
        rag_config: Optional RAG configuration
        prompt: Optional prompt for structured information extraction
        use_medical_templates: Whether to use automatic medical template selection
        generate_cue_cards: Whether to generate cue cards from LLM output
        context_type: Context type for cue cards (medical, situational, clinical)
    """
    # Import config
    from .config import get_config, RAGConfig
    
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

    # Medical template selection and analysis
    analysis_results = {}
    cue_cards_results = {}
    template_info = {}
    
    if use_medical_templates and not prompt:
        # Automatic medical template selection
        if config.verbose:
            print("Analyzing document type and selecting optimal templates...")
        
        # Analyze document type using vector similarity
        doc_type_scores = analyze_document_type(parsed)
        if config.verbose:
            print(f"Document type analysis: {doc_type_scores}")
        
        # Select optimal templates using vector similarity
        task_description = "Analyze medical document and create comprehensive documentation"
        optimal_templates = select_optimal_templates(parsed, task_description, 
                                                    similarity_threshold=0.3, top_k=3)
        
        if optimal_templates:
            if config.verbose:
                print(f"Selected templates: {optimal_templates}")
            
            # Use the top recommended template
            template_name, confidence = optimal_templates[0]
            template = get_template(template_name)
            template_info = {
                'template_name': template_name,
                'confidence': confidence,
                'all_recommendations': optimal_templates
            }
            
            if config.verbose:
                print(f"Using template: {template_name} (confidence: {confidence:.2f})")
            
            # Create a copy of config and update with selected template
            if rag_config is None:
                # Make a copy of the default config
                rag_config = deepcopy(config)
            
            rag_config.use_prompt_template = True
            rag_config.custom_template = template
            
            # Analyze with selected template
            results = analyze_with_llm(parsed, references, config=rag_config)
        else:
            if config.verbose:
                print("No suitable medical templates found, using default analysis")
            results = analyze_with_llm(parsed, references, config=rag_config)
    else:
        # Standard analysis (with custom prompt or template disabled)
        if prompt:
            results = analyze_with_llm(parsed, prompt=prompt, config=rag_config)
        else:
            results = analyze_with_llm(parsed, references, config=rag_config)
    
    # Generate cue cards using clustering
    if generate_cue_cards:
        if config.verbose:
            print("Extracting cue cards using clustering...")
        
        if isinstance(results, dict):
            for key, content in results.items():
                cue_cards = extract_cue_cards(content, context_type=context_type)
                cue_cards_results[key] = {
                    'cue_cards': cue_cards
                }
        else:
            cue_cards = extract_cue_cards(results, context_type=context_type)
            cue_cards_results['main'] = {
                'cue_cards': cue_cards, 
            }
    
    # Compile comprehensive results
    final_results = {
        'analysis': results,
        'template_info': template_info,
        'cue_cards': cue_cards_results,
        'document_type_scores': doc_type_scores if use_medical_templates else None
    }
    
    return final_results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process documents using Gemma RAG with medical templates and cue cards")
    parser.add_argument("--file", default="Improved Specificity Clusters.pdf", 
                       help="Path to document file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--template", help="Override default prompt template")
    parser.add_argument("--no-template", action="store_true",
                       help="Disable prompt templates (use basic prompts)")
    parser.add_argument("--no-references", action="store_true", 
                       help="Skip reference document analysis")
    parser.add_argument("--no-medical-templates", action="store_true",
                       help="Disable automatic medical template selection")
    parser.add_argument("--no-cue-cards", action="store_true",
                       help="Disable cue card generation")
    parser.add_argument("--context-type", default="medical", 
                       choices=["medical", "situational", "clinical"],
                       help="Context type for cue cards (default: medical)")
    parser.add_argument("--prompt", help="Custom prompt for analysis")
    args = parser.parse_args()
    
    # Load your corpus of reference documents as raw text
    references = ["reference doc 1...", "reference doc 2...", "..."] if not args.no_references else []
    meta = [{"doc_id": "DocA"}, {"doc_id": "DocB"}, {}] if not args.no_references else None
    
    # Create config if options provided
    config_needed = (args.verbose or args.template or args.no_template or 
                    args.no_references or args.no_medical_templates or args.no_cue_cards)
    
    if config_needed:
        from .config import RAGConfig
        config = RAGConfig()
        if args.verbose:
            config.verbose = True
        if args.template:
            config.default_template = args.template
        if args.no_template:
            config.use_prompt_template = False
        if args.no_references:
            config.include_references = False
            
        # Process with new options
        output = process_document(
            args.file, 
            references, 
            meta, 
            config,
            prompt=args.prompt,
            use_medical_templates=not args.no_medical_templates,
            generate_cue_cards=not args.no_cue_cards,
            context_type=args.context_type
        )
    else:
        # Use default config with new features enabled
        output = process_document(
            args.file, 
            references, 
            meta,
            prompt=args.prompt,
            use_medical_templates=True,
            generate_cue_cards=True,
            context_type=args.context_type
        )

    # Save comprehensive output with new structure
    if isinstance(output, dict) and 'analysis' in output:
        # New enhanced structure with medical templates and cue cards
        analysis = output.get('analysis', {})
        template_info = output.get('template_info', {})
        cue_cards = output.get('cue_cards', {})
        doc_scores = output.get('document_type_scores', {})
        
        # Save main analysis
        if isinstance(analysis, dict):
            # Multiple reference analyses
            for key, content in analysis.items():
                output_file = f'report_{key}.txt'
                with open(output_file, 'w') as file:
                    file.write(content)
                print(f"Analysis report saved to: {output_file}")
        else:
            # Single analysis
            output_file = 'report_analysis.txt'
            with open(output_file, 'w') as file:
                file.write(str(analysis))
            print(f"Analysis report saved to: {output_file}")
        
        # Save cue cards
        if cue_cards:
            for key, card_data in cue_cards.items():
                cue_cards_file = f'cue_cards_{key}.txt'
                with open(cue_cards_file, 'w') as file:
                    # Format cue cards inline
                    formatted_cards = '\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in card_data['cue_cards']])
                    file.write(formatted_cards)
                print(f"Cue cards saved to: {cue_cards_file}")
        
        # Save template information
        if template_info:
            template_file = 'template_info.json'
            with open(template_file, 'w') as file:
                json.dump(template_info, file, indent=2)
            print(f"Template information saved to: {template_file}")
        
        # Save document analysis scores
        if doc_scores:
            scores_file = 'document_type_analysis.json'
            with open(scores_file, 'w') as file:
                json.dump(doc_scores, file, indent=2)
            print(f"Document type analysis saved to: {scores_file}")
        
        # Create comprehensive combined report
        combined_file = 'comprehensive_report.txt'
        with open(combined_file, 'w') as file:
            file.write("COMPREHENSIVE MEDICAL DOCUMENT ANALYSIS\n")
            file.write("=" * 60 + "\n\n")
            
            # Template information
            if template_info:
                file.write("TEMPLATE SELECTION\n")
                file.write("-" * 20 + "\n")
                file.write(f"Selected Template: {template_info.get('template_name', 'N/A')}\n")
                file.write(f"Confidence: {template_info.get('confidence', 0):.2f}\n")
                if 'all_recommendations' in template_info:
                    file.write("All Recommendations:\n")
                    for name, conf in template_info['all_recommendations']:
                        file.write(f"  - {name}: {conf:.2f}\n")
                file.write("\n")
            
            # Document type analysis
            if doc_scores:
                file.write("DOCUMENT TYPE ANALYSIS\n")
                file.write("-" * 25 + "\n")
                for doc_type, score in doc_scores.items():
                    file.write(f"{doc_type.capitalize()}: {score:.3f}\n")
                file.write("\n")
            
            # Main analysis
            file.write("ANALYSIS RESULTS\n")
            file.write("-" * 20 + "\n")
            if isinstance(analysis, dict):
                for key, content in analysis.items():
                    file.write(f"\n{key.upper()}\n")
                    file.write("=" * len(key) + "\n")
                    file.write(str(content))
                    file.write("\n\n")
            else:
                file.write(str(analysis))
                file.write("\n\n")
            
            # Cue cards
            if cue_cards:
                file.write("CUE CARDS SUMMARY\n")
                file.write("-" * 20 + "\n")
                for key, card_data in cue_cards.items():
                    file.write(f"\n{key.upper()} CONTEXT:\n")
                    formatted_cards = '\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in card_data['cue_cards']])
                    file.write(formatted_cards)
                    file.write("\n")
        
        print(f"\nComprehensive report saved to: {combined_file}")
        
        # Display summary
        print("\n" + "="*60)
        print("ENHANCED MEDICAL ANALYSIS COMPLETE")
        print("="*60)
        if template_info:
            print(f"Template Used: {template_info.get('template_name', 'N/A')} "
                  f"(confidence: {template_info.get('confidence', 0):.2f})")
        if cue_cards:
            total_cards = sum(len(card_data['cue_cards']) for card_data in cue_cards.values())
            print(f"Generated {total_cards} cue cards across {len(cue_cards)} contexts")
        if isinstance(analysis, dict):
            print(f"Generated {len(analysis)} reference comparisons")
        
        # Show sample cue cards if available
        if cue_cards and args.verbose:
            print("\nSAMPLE CUE CARDS:")
            print("-" * 20)
            first_key = next(iter(cue_cards))
            sample_cards = '\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in cue_cards[first_key]['cue_cards']])
            # Show first few lines
            lines = sample_cards.split('\n')[:15]
            print('\n'.join(lines))
            if len(sample_cards.split('\n')) > 15:
                print("... (see cue card files for complete output)")
            
    else:
        # Fallback for old format or simple string output
        output_file = 'report.txt'
        with open(output_file, 'w') as file:
            file.write(str(output))
        print(f"\nReport saved to: {output_file}")
        print("\n" + "="*50)
        print("ANALYSIS REPORT")
        print("="*50)
        print(str(output))

