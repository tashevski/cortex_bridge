"""
Prompt Templates for Gemma RAG System
====================================

This module provides various prompt templates that can be used with the Gemma client's
prompt_template parameter. Templates use {context} and {prompt} placeholders.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Container for a prompt template with metadata"""
    name: str
    description: str
    template: str
    best_for: List[str]  # List of use cases this template is best for

# Analysis Templates
ANALYSIS_TEMPLATES = {
    "structured_analysis": PromptTemplate(
        name="Structured Analysis",
        description="Comprehensive structured analysis of documents",
        template="""<start_of_turn>system
You are an expert document analyst. Analyze the provided content systematically.
<end_of_turn>

<start_of_turn>user
Context: {context}

Task: {prompt}

Please provide a structured analysis including:
1. Main topics and themes
2. Key findings or claims
3. Supporting evidence
4. Implications
5. Recommendations or conclusions
<end_of_turn>

<start_of_turn>model""",
        best_for=["research papers", "reports", "technical documents"]
    ),
    
    "executive_summary": PromptTemplate(
        name="Executive Summary",
        description="Creates concise executive summaries",
        template="""<start_of_turn>system
You are an expert at creating clear, concise executive summaries.
<end_of_turn>

<start_of_turn>user
Document Context: {context}

Request: {prompt}

Create an executive summary that includes:
- Key Points (3-5 bullets)
- Main Findings
- Critical Insights
- Recommended Actions

Keep it concise and actionable.
<end_of_turn>

<start_of_turn>model""",
        best_for=["business documents", "long reports", "decision making"]
    ),
    
    "comparative_analysis": PromptTemplate(
        name="Comparative Analysis",
        description="Compares document content with reference materials",
        template="""<start_of_turn>system
You are an expert at comparative analysis. Compare and contrast the provided materials.
<end_of_turn>

<start_of_turn>user
Primary Document: {context}

Analysis Request: {prompt}

Provide a comparative analysis that:
1. Identifies similarities and differences
2. Highlights unique contributions
3. Assesses relative strengths/weaknesses
4. Synthesizes insights across documents
<end_of_turn>

<start_of_turn>model""",
        best_for=["literature reviews", "competitive analysis", "research comparison"]
    )
}

# Question-Answering Templates
QA_TEMPLATES = {
    "detailed_qa": PromptTemplate(
        name="Detailed Q&A",
        description="Provides comprehensive answers with evidence",
        template="""<start_of_turn>system
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
        best_for=["research queries", "fact-finding", "detailed explanations"]
    ),
    
    "concise_qa": PromptTemplate(
        name="Concise Q&A",
        description="Brief, direct answers",
        template="""<start_of_turn>system
Provide brief, accurate answers based on the context.
<end_of_turn>

<start_of_turn>user
Context: {context}

Question: {prompt}

Answer concisely in 1-3 sentences.
<end_of_turn>

<start_of_turn>model""",
        best_for=["quick queries", "fact checking", "yes/no questions"]
    )
}

# Extraction Templates
EXTRACTION_TEMPLATES = {
    "entity_extraction": PromptTemplate(
        name="Entity Extraction",
        description="Extracts specific entities from documents",
        template="""<start_of_turn>system
You are an expert at identifying and extracting entities from text.
<end_of_turn>

<start_of_turn>user
Document: {context}

Task: {prompt}

Extract and categorize all relevant entities:
- People/Organizations
- Dates/Times
- Locations
- Key Concepts/Terms
- Numerical Data
- Relationships

Format as structured lists.
<end_of_turn>

<start_of_turn>model""",
        best_for=["data mining", "information extraction", "database population"]
    ),
    
    "key_points": PromptTemplate(
        name="Key Points Extraction",
        description="Extracts main points and arguments",
        template="""<start_of_turn>system
Extract the most important points from the provided content.
<end_of_turn>

<start_of_turn>user
Content: {context}

Request: {prompt}

Extract:
• Main arguments or claims
• Supporting evidence
• Conclusions
• Action items (if any)

Present as bullet points.
<end_of_turn>

<start_of_turn>model""",
        best_for=["meeting notes", "article summaries", "quick reviews"]
    )
}

# Generation Templates
GENERATION_TEMPLATES = {
    "report_generation": PromptTemplate(
        name="Report Generation",
        description="Generates formal reports from document analysis",
        template="""<start_of_turn>system
You are a professional report writer. Create well-structured reports based on the provided analysis.
<end_of_turn>

<start_of_turn>user
Analysis Results: {context}

Report Requirements: {prompt}

Generate a professional report with:
1. Executive Summary
2. Introduction
3. Methodology (if applicable)
4. Findings
5. Analysis
6. Recommendations
7. Conclusion

Use formal language and clear structure.
<end_of_turn>

<start_of_turn>model""",
        best_for=["formal reports", "research outputs", "professional documentation"]
    ),
    
    "insight_generation": PromptTemplate(
        name="Insight Generation",
        description="Generates actionable insights from data",
        template="""<start_of_turn>system
You are an insight generator. Transform data into actionable insights.
<end_of_turn>

<start_of_turn>user
Data/Context: {context}

Focus Area: {prompt}

Generate insights that are:
- Actionable
- Evidence-based
- Strategic
- Forward-looking

Include:
1. Key Insights (3-5)
2. Supporting Data
3. Implications
4. Recommended Next Steps
<end_of_turn>

<start_of_turn>model""",
        best_for=["business intelligence", "strategy planning", "decision support"]
    )
}

# Specialized Templates
SPECIALIZED_TEMPLATES = {
    "technical_review": PromptTemplate(
        name="Technical Review",
        description="Reviews technical content and documentation",
        template="""<start_of_turn>system
You are a technical reviewer with expertise in analyzing technical documentation.
<end_of_turn>

<start_of_turn>user
Technical Content: {context}

Review Focus: {prompt}

Provide a technical review covering:
1. Technical accuracy
2. Completeness
3. Clarity and organization
4. Best practices adherence
5. Potential improvements
6. Risk factors or concerns
<end_of_turn>

<start_of_turn>model""",
        best_for=["code documentation", "technical specs", "architecture reviews"]
    ),
    
    "academic_critique": PromptTemplate(
        name="Academic Critique",
        description="Provides scholarly critique of academic content",
        template="""<start_of_turn>system
You are an academic reviewer providing scholarly critique.
<end_of_turn>

<start_of_turn>user
Academic Content: {context}

Critique Focus: {prompt}

Provide a scholarly critique examining:
1. Thesis and arguments
2. Methodology (if applicable)
3. Evidence quality
4. Literature integration
5. Logical consistency
6. Contribution to field
7. Limitations and gaps
<end_of_turn>

<start_of_turn>model""",
        best_for=["peer review", "thesis review", "research papers"]
    )
}

# Combine all templates
ALL_TEMPLATES = {
    **ANALYSIS_TEMPLATES,
    **QA_TEMPLATES,
    **EXTRACTION_TEMPLATES,
    **GENERATION_TEMPLATES,
    **SPECIALIZED_TEMPLATES
}

def get_template(name: str) -> Optional[str]:
    """Get a template by name, returns the template string or None"""
    template_obj = ALL_TEMPLATES.get(name)
    return template_obj.template if template_obj else None

def get_template_info(name: str) -> Optional[PromptTemplate]:
    """Get full template information including metadata"""
    return ALL_TEMPLATES.get(name)

def list_templates(category: Optional[str] = None) -> Dict[str, PromptTemplate]:
    """List all templates or templates from a specific category"""
    if category is None:
        return ALL_TEMPLATES
    
    categories = {
        "analysis": ANALYSIS_TEMPLATES,
        "qa": QA_TEMPLATES,
        "extraction": EXTRACTION_TEMPLATES,
        "generation": GENERATION_TEMPLATES,
        "specialized": SPECIALIZED_TEMPLATES
    }
    
    return categories.get(category.lower(), {})

def get_template_for_use_case(use_case: str) -> List[str]:
    """Get template names best suited for a specific use case"""
    use_case_lower = use_case.lower()
    suitable_templates = []
    
    for name, template in ALL_TEMPLATES.items():
        for best_for in template.best_for:
            if use_case_lower in best_for.lower():
                suitable_templates.append(name)
                break
    
    return suitable_templates

def create_custom_template(context_prefix: str = "", 
                         prompt_prefix: str = "",
                         instructions: str = "",
                         output_format: str = "") -> str:
    """Create a custom template with specified components"""
    template = "<start_of_turn>system\n"
    
    if instructions:
        template += f"{instructions}\n"
    else:
        template += "You are a helpful AI assistant.\n"
    
    template += "<end_of_turn>\n\n<start_of_turn>user\n"
    
    if context_prefix:
        template += f"{context_prefix}: {{context}}\n\n"
    else:
        template += "Context: {context}\n\n"
    
    if prompt_prefix:
        template += f"{prompt_prefix}: {{prompt}}\n"
    else:
        template += "Task: {prompt}\n"
    
    if output_format:
        template += f"\n{output_format}\n"
    
    template += "<end_of_turn>\n\n<start_of_turn>model"
    
    return template

# Example usage with Gemma client
def get_example_usage():
    """Show how to use templates with the Gemma client"""
    example = """
# Using with GemmaClient
from ai.gemma_client import GemmaClient
from rag_functions.prompt_templates import get_template

client = GemmaClient()

# Get a template
template = get_template("structured_analysis")

# Use it with the client
response = client.generate_response(
    prompt="Analyze the main themes in this document",
    context=document_text,
    prompt_template=template
)

# Or use a custom template
custom = create_custom_template(
    context_prefix="Document to analyze",
    prompt_prefix="Analysis request",
    instructions="You are an expert analyst. Be thorough and insightful.",
    output_format="Format your response with clear headings and bullet points."
)

response = client.generate_response(
    prompt="What are the key insights?",
    context=document_text,
    prompt_template=custom
)
"""
    return example

if __name__ == "__main__":
    # Print available templates
    print("Available Prompt Templates:")
    print("=" * 50)
    
    for category in ["analysis", "qa", "extraction", "generation", "specialized"]:
        print(f"\n{category.upper()} TEMPLATES:")
        templates = list_templates(category)
        for name, template in templates.items():
            print(f"  - {name}: {template.description}")
            print(f"    Best for: {', '.join(template.best_for)}")
    
    print("\n" + "=" * 50)
    print("Example Usage:")
    print(get_example_usage())