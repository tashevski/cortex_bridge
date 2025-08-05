#!/usr/bin/env python3
"""
Example Usage of Prompt Templates with RAG System
================================================

This script demonstrates how to use different prompt templates
with the Gemma RAG system for various analysis tasks.
"""

import sys
from pathlib import Path

# Add program_files to path
sys.path.append(str(Path(__file__).parent.parent / "program_files"))

from rag_functions.core.config import get_config, RAGConfig
from rag_functions.templates.prompt_templates import get_template, list_templates, get_template_for_use_case, create_custom_template
from rag_functions.core.main import process_document

def demo_available_templates():
    """Show all available templates"""
    print("🎯 Available Prompt Templates")
    print("=" * 50)
    
    categories = ["analysis", "qa", "extraction", "generation", "specialized"]
    
    for category in categories:
        templates = list_templates(category)
        if templates:
            print(f"\n📋 {category.upper()} Templates:")
            for name, template_info in templates.items():
                print(f"  • {name}")
                print(f"    └─ {template_info.description}")
                print(f"    └─ Best for: {', '.join(template_info.best_for)}")

def demo_template_usage():
    """Demonstrate how to use templates with different configurations"""
    
    print("\n🚀 Template Usage Examples")
    print("=" * 50)
    
    # Example 1: Using predefined template with config
    print("\n1️⃣ Using Structured Analysis Template:")
    config = get_config("quality")  # Uses structured_analysis by default
    print(f"   Config: {config.default_template}")
    
    # Example 2: Override template in config
    print("\n2️⃣ Overriding Template:")
    config = get_config("balanced")
    config.default_template = "executive_summary"
    print(f"   Overridden template: {config.default_template}")
    
    # Example 3: Using custom template
    print("\n3️⃣ Creating Custom Template:")
    custom_template = create_custom_template(
        context_prefix="Research Data",
        prompt_prefix="Analysis Task",
        instructions="You are a research analyst specializing in data interpretation.",
        output_format="Provide your analysis in numbered sections with clear headings."
    )
    config = get_config("balanced")
    config.custom_template = custom_template
    print("   Created custom template with specialized instructions")

def demo_use_case_matching():
    """Show how to find templates for specific use cases"""
    
    print("\n🎯 Template Recommendations by Use Case")
    print("=" * 50)
    
    use_cases = [
        "research papers",
        "business documents",
        "technical documents",
        "quick queries",
        "fact checking"
    ]
    
    for use_case in use_cases:
        templates = get_template_for_use_case(use_case)
        print(f"\n📄 {use_case.title()}:")
        if templates:
            for template in templates:
                print(f"  • {template}")
        else:
            print("  • No specific templates found (use general templates)")

def demo_template_content():
    """Show actual template content"""
    
    print("\n📝 Template Content Examples")
    print("=" * 50)
    
    examples = ["structured_analysis", "concise_qa", "entity_extraction"]
    
    for template_name in examples:
        template = get_template(template_name)
        if template:
            print(f"\n📋 {template_name.replace('_', ' ').title()}:")
            print("-" * 30)
            # Show first few lines
            lines = template.split('\n')[:8]
            for line in lines:
                print(f"  {line}")
            if len(template.split('\n')) > 8:
                print("  ...")

def demo_practical_example():
    """Show a practical example of using templates"""
    
    print("\n💡 Practical Example")
    print("=" * 50)
    
    # Create a sample document scenario
    sample_references = [
        "Sample reference document about AI ethics...",
        "Another reference about machine learning applications..."
    ]
    
    # Example configurations for different analysis types
    scenarios = [
        {
            "name": "Quick Summary",
            "config": RAGConfig(
                default_template="executive_summary",
                detailed_report=False,
                request_timeout=30
            )
        },
        {
            "name": "Detailed Analysis", 
            "config": RAGConfig(
                default_template="structured_analysis",
                detailed_report=True,
                request_timeout=90
            )
        },
        {
            "name": "Entity Extraction",
            "config": RAGConfig(
                default_template="entity_extraction",
                detailed_report=False,
                request_timeout=60
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔍 Scenario: {scenario['name']}")
        print(f"   Template: {scenario['config'].default_template}")
        print(f"   Timeout: {scenario['config'].request_timeout}s")
        print(f"   Detailed: {scenario['config'].detailed_report}")
        
        # In a real scenario, you would call:
        # result = process_document("document.pdf", sample_references, config=scenario['config'])

def demo_custom_template_creation():
    """Show how to create custom templates for specific needs"""
    
    print("\n🛠️ Custom Template Creation")
    print("=" * 50)
    
    # Example: Custom template for financial analysis
    financial_template = create_custom_template(
        context_prefix="Financial Document",
        prompt_prefix="Analysis Request",
        instructions="You are a financial analyst. Focus on numerical data, trends, and financial implications.",
        output_format="""Structure your analysis as follows:
1. Executive Summary
2. Key Financial Metrics
3. Trend Analysis
4. Risks Assessment
5. Recommendations"""
    )
    
    print("\n📊 Financial Analysis Template:")
    print("   ✓ Custom context prefix")
    print("   ✓ Specialized instructions")
    print("   ✓ Structured output format")
    
    # Example: Custom template for code review
    code_review_template = create_custom_template(
        context_prefix="Code/Documentation",
        prompt_prefix="Review Focus",
        instructions="You are a senior software engineer conducting a code review. Focus on quality, maintainability, and best practices.",
        output_format="""Provide review in sections:
- Code Quality Assessment
- Security Considerations  
- Performance Notes
- Maintainability Score
- Recommendations"""
    )
    
    print("\n💻 Code Review Template:")
    print("   ✓ Technical focus")
    print("   ✓ Structured evaluation")
    print("   ✓ Action-oriented output")

def main():
    """Run all demonstrations"""
    
    print("🎨 Prompt Templates for Gemma RAG System")
    print("=" * 60)
    
    try:
        demo_available_templates()
        demo_use_case_matching()
        demo_template_usage()
        demo_template_content()
        demo_practical_example()
        demo_custom_template_creation()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\n💡 To use templates in your code:")
        print("   1. Import: from rag_functions.prompt_templates import get_template")
        print("   2. Get template: template = get_template('structured_analysis')")
        print("   3. Use with Gemma: client.generate_response(prompt, context, prompt_template=template)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all required modules are available.")

if __name__ == "__main__":
    main()