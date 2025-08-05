"""
Medical Documentation Prompt Templates
=====================================

This module provides specialized prompt templates for medical documentation use cases
with the Gemma client's prompt_template parameter. Templates use {context} and {prompt} placeholders.

Templates are designed for:
- Patient intake and history documentation
- Clinical note generation and analysis
- Medical report creation
- Diagnostic analysis and decision support
- Treatment planning and care coordination
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

# Patient Intake Templates
INTAKE_TEMPLATES = {
    "comprehensive_intake": PromptTemplate(
        name="Comprehensive Patient Intake",
        description="Structures comprehensive patient intake documentation",
        template="""<start_of_turn>system
You are a medical documentation specialist. Create comprehensive patient intake documentation from provided information.
<end_of_turn>

<start_of_turn>user
Patient Information: {context}

Documentation Request: {prompt}

Create a comprehensive intake summary including:
1. Chief Complaint
2. History of Present Illness (HPI)
3. Past Medical History (PMH)
4. Medications and Allergies
5. Social History
6. Family History
7. Review of Systems (ROS)
8. Assessment and Plan

Use standard medical terminology and SOAP format where appropriate.
<end_of_turn>

<start_of_turn>model""",
        best_for=["new patient visits", "annual physicals", "intake documentation"]
    ),
    
    "symptom_assessment": PromptTemplate(
        name="Symptom Assessment",
        description="Analyzes and documents patient symptoms systematically",
        template="""<start_of_turn>system
You are a clinical assessment specialist. Analyze patient symptoms using standardized medical assessment criteria.
<end_of_turn>

<start_of_turn>user
Patient Presentation: {context}

Assessment Focus: {prompt}

Provide systematic symptom assessment:
- Onset, Location, Duration, Character (OLDCART)
- Associated symptoms
- Aggravating/alleviating factors
- Severity scale (1-10)
- Functional impact
- Red flag symptoms
- Differential considerations

Format using clinical assessment standards.
<end_of_turn>

<start_of_turn>model""",
        best_for=["symptom documentation", "triage assessments", "clinical evaluations"]
    ),
    
    "medical_history": PromptTemplate(
        name="Medical History Documentation",
        description="Structures and analyzes patient medical history",
        template="""<start_of_turn>system
You are a medical historian specializing in comprehensive patient history documentation.
<end_of_turn>

<start_of_turn>user
Patient History Data: {context}

Documentation Task: {prompt}

Create structured medical history including:
1. Past Medical History
   - Chronic conditions
   - Previous hospitalizations
   - Surgical history
2. Medication History
   - Current medications with dosages
   - Previous medication trials
   - Adverse drug reactions
3. Family History
   - Genetic predispositions
   - Hereditary conditions
4. Social History
   - Lifestyle factors
   - Occupational exposures
   - Risk behaviors

Highlight clinically significant findings.
<end_of_turn>

<start_of_turn>model""",
        best_for=["comprehensive histories", "chronic disease management", "specialist referrals"]
    )
}

# Clinical Note Templates
CLINICAL_TEMPLATES = {
    "soap_note": PromptTemplate(
        name="SOAP Note Generation",
        description="Creates structured SOAP (Subjective, Objective, Assessment, Plan) notes",
        template="""<start_of_turn>system
You are a clinical documentation specialist. Create accurate SOAP notes from patient encounter information.
<end_of_turn>

<start_of_turn>user
Patient Encounter Data: {context}

Documentation Request: {prompt}

Generate a SOAP note with:

SUBJECTIVE:
- Chief complaint
- History of present illness
- Review of systems
- Patient's perspective

OBJECTIVE:
- Vital signs
- Physical examination findings
- Laboratory/diagnostic results
- Mental status

ASSESSMENT:
- Clinical impression
- Differential diagnosis
- Problem list with ICD-10 codes

PLAN:
- Treatment recommendations
- Medications prescribed
- Follow-up instructions
- Patient education provided

Use appropriate medical terminology and documentation standards.
<end_of_turn>

<start_of_turn>model""",
        best_for=["routine visits", "follow-up appointments", "clinical documentation"]
    ),
    
    "progress_note": PromptTemplate(
        name="Progress Note Documentation",
        description="Documents patient progress and ongoing care",
        template="""<start_of_turn>system
You are a healthcare provider documenting patient progress. Focus on changes since last visit and ongoing care management.
<end_of_turn>

<start_of_turn>user
Patient Progress Information: {context}

Progress Documentation: {prompt}

Create progress note including:
1. Interval History
   - Changes since last visit
   - Symptom progression
   - Medication compliance
2. Current Status
   - Present symptoms
   - Functional status
   - Quality of life measures
3. Objective Findings
   - Physical exam changes
   - Recent test results
4. Assessment of Progress
   - Response to treatment
   - Goal achievement
5. Plan Modifications
   - Treatment adjustments
   - New interventions
   - Next steps

Emphasize clinical decision-making rationale.
<end_of_turn>

<start_of_turn>model""",
        best_for=["chronic disease management", "treatment monitoring", "follow-up care"]
    ),
    
    "discharge_summary": PromptTemplate(
        name="Discharge Summary",
        description="Creates comprehensive discharge summaries",
        template="""<start_of_turn>system
You are a hospital physician creating a discharge summary. Ensure continuity of care through comprehensive documentation.
<end_of_turn>

<start_of_turn>user
Hospitalization Data: {context}

Discharge Summary Request: {prompt}

Create discharge summary with:

ADMISSION INFORMATION:
- Date of admission
- Admitting diagnosis
- Chief complaint

HOSPITAL COURSE:
- Major interventions
- Complications
- Response to treatment
- Significant events

DISCHARGE STATUS:
- Final diagnoses
- Condition at discharge
- Functional status

MEDICATIONS:
- Discharge medications
- Changes from admission
- Medication reconciliation

FOLLOW-UP CARE:
- Appointments scheduled
- Continuing care needs
- Activity restrictions
- Warning signs to report

Include clear recommendations for ongoing care providers.
<end_of_turn>

<start_of_turn>model""",
        best_for=["hospital discharges", "care transitions", "specialist referrals"]
    )
}

# Medical Report Templates
REPORT_TEMPLATES = {
    "diagnostic_report": PromptTemplate(
        name="Diagnostic Report Generation",
        description="Creates comprehensive diagnostic reports from test results and clinical data",
        template="""<start_of_turn>system
You are a medical diagnostician creating comprehensive diagnostic reports. Integrate clinical data with evidence-based medicine.
<end_of_turn>

<start_of_turn>user
Clinical Data: {context}

Report Request: {prompt}

Generate diagnostic report with:

CLINICAL PRESENTATION:
- Patient demographics
- Presenting symptoms
- Clinical timeline

DIAGNOSTIC WORKUP:
- Laboratory results
- Imaging findings
- Procedure results
- Special tests

CLINICAL CORRELATION:
- Significant findings
- Normal variants
- Incidental findings

DIAGNOSTIC IMPRESSION:
- Primary diagnosis
- Differential diagnoses
- Confidence levels
- Supporting evidence

RECOMMENDATIONS:
- Additional testing needed
- Treatment considerations
- Follow-up requirements
- Specialist referrals

Use evidence-based diagnostic criteria and include relevant clinical guidelines.
<end_of_turn>

<start_of_turn>model""",
        best_for=["diagnostic workups", "test result interpretation", "clinical decision support"]
    ),
    
    "pathology_report": PromptTemplate(
        name="Pathology Report Analysis",
        description="Analyzes and summarizes pathology reports for clinical correlation",
        template="""<start_of_turn>system
You are a pathologist providing clinical correlation of pathology findings. Focus on clinically relevant implications.
<end_of_turn>

<start_of_turn>user
Pathology Data: {context}

Analysis Request: {prompt}

Provide pathology analysis with:

SPECIMEN INFORMATION:
- Specimen type and site
- Clinical indication
- Gross description

MICROSCOPIC FINDINGS:
- Histological features
- Cellular characteristics
- Architectural patterns
- Special stains results

DIAGNOSIS:
- Final pathological diagnosis
- Grade/stage if applicable
- Margin status
- Molecular markers

CLINICAL SIGNIFICANCE:
- Prognostic implications
- Treatment relevance
- Follow-up recommendations
- Correlation with clinical presentation

Include relevant staging systems and classification criteria.
<end_of_turn>

<start_of_turn>model""",
        best_for=["pathology review", "oncology reports", "surgical planning"]
    ),
    
    "radiology_report": PromptTemplate(
        name="Radiology Report Interpretation",
        description="Interprets and correlates radiology reports with clinical context",
        template="""<start_of_turn>system
You are a radiologist providing clinical correlation of imaging studies. Emphasize clinically actionable findings.
<end_of_turn>

<start_of_turn>user
Imaging Data: {context}

Interpretation Request: {prompt}

Provide radiology interpretation with:

STUDY INFORMATION:
- Imaging modality
- Clinical indication
- Technique/protocol

FINDINGS:
- Relevant positive findings
- Pertinent negatives
- Comparison with prior studies
- Measurements and quantification

IMPRESSION:
- Primary findings
- Differential considerations
- Confidence assessment
- Limitations of study

CLINICAL CORRELATION:
- Relationship to symptoms
- Impact on diagnosis
- Treatment implications
- Recommended follow-up

Use standardized reporting terminology (BI-RADS, LI-RADS, etc.) when applicable.
<end_of_turn>

<start_of_turn>model""",
        best_for=["imaging interpretation", "diagnostic correlation", "multidisciplinary planning"]
    )
}

# Diagnostic Analysis Templates
DIAGNOSTIC_TEMPLATES = {
    "differential_diagnosis": PromptTemplate(
        name="Differential Diagnosis Analysis",
        description="Systematically analyzes symptoms and findings to generate differential diagnoses",
        template="""<start_of_turn>system
You are a clinical diagnostician. Use systematic diagnostic reasoning to analyze patient presentations and generate evidence-based differential diagnoses.
<end_of_turn>

<start_of_turn>user
Clinical Presentation: {context}

Diagnostic Challenge: {prompt}

Provide systematic diagnostic analysis:

CLINICAL SYNTHESIS:
- Key symptoms and signs
- Relevant history
- Physical findings
- Laboratory/imaging abnormalities

DIFFERENTIAL DIAGNOSIS:
1. Most likely diagnosis
   - Supporting evidence
   - Diagnostic criteria met
2. Alternative diagnoses (ranked by likelihood)
   - Supporting/opposing evidence
   - Missing criteria
3. Must-not-miss diagnoses
   - Red flag symptoms
   - Risk factors present

DIAGNOSTIC WORKUP:
- Additional tests needed
- Sequence of testing
- Cost-effectiveness considerations

CLINICAL REASONING:
- Pattern recognition used
- Diagnostic uncertainty areas
- Decision-making rationale

Include relevant clinical guidelines and diagnostic criteria.
<end_of_turn>

<start_of_turn>model""",
        best_for=["complex diagnoses", "medical education", "clinical decision support"]
    ),
    
    "case_analysis": PromptTemplate(
        name="Clinical Case Analysis",
        description="Comprehensive analysis of clinical cases for educational or review purposes",
        template="""<start_of_turn>system
You are a medical educator analyzing clinical cases. Provide comprehensive educational analysis emphasizing learning points.
<end_of_turn>

<start_of_turn>user
Clinical Case: {context}

Analysis Focus: {prompt}

Provide comprehensive case analysis:

CASE SUMMARY:
- Patient presentation
- Key timeline events
- Clinical decisions made

DIAGNOSTIC REASONING:
- Initial differential
- Diagnostic process
- Test interpretation
- Final diagnosis

MANAGEMENT ANALYSIS:
- Treatment decisions
- Response to therapy
- Complications encountered
- Outcome achieved

LEARNING POINTS:
- Key clinical pearls
- Diagnostic pitfalls avoided/encountered
- Management considerations
- Evidence-based practices demonstrated

DISCUSSION QUESTIONS:
- Alternative approaches
- What-if scenarios
- Areas for improvement
- Broader clinical implications

Format for medical education and case-based learning.
<end_of_turn>

<start_of_turn>model""",
        best_for=["medical education", "case reviews", "quality improvement"]
    ),
    
    "clinical_decision_support": PromptTemplate(
        name="Clinical Decision Support",
        description="Provides evidence-based decision support for clinical scenarios",
        template="""<start_of_turn>system
You are a clinical decision support system. Provide evidence-based recommendations while acknowledging clinical judgment requirements.
<end_of_turn>

<start_of_turn>user
Clinical Scenario: {context}

Decision Support Request: {prompt}

Provide clinical decision support with:

CLINICAL SITUATION:
- Current patient status
- Decision point reached
- Available options

EVIDENCE REVIEW:
- Relevant clinical guidelines
- Research evidence quality
- Strength of recommendations
- Population applicability

RISK-BENEFIT ANALYSIS:
- Potential benefits of each option
- Associated risks/complications
- Patient-specific factors
- Contraindications

RECOMMENDATIONS:
- Preferred approach with rationale
- Alternative options
- Monitoring requirements
- Contingency planning

CLINICAL CONSIDERATIONS:
- Patient preferences integration
- Resource availability
- Timing factors
- Follow-up needs

Always emphasize that clinical judgment and patient-specific factors must guide final decisions.
<end_of_turn>

<start_of_turn>model""",
        best_for=["treatment decisions", "clinical protocols", "evidence-based practice"]
    )
}

# Treatment Planning Templates
TREATMENT_TEMPLATES = {
    "treatment_plan": PromptTemplate(
        name="Comprehensive Treatment Planning",
        description="Creates detailed, evidence-based treatment plans",
        template="""<start_of_turn>system
You are a healthcare provider creating comprehensive treatment plans. Integrate evidence-based medicine with patient-centered care principles.
<end_of_turn>

<start_of_turn>user
Patient Clinical Data: {context}

Treatment Planning Request: {prompt}

Develop comprehensive treatment plan with:

PROBLEM LIST:
- Primary diagnoses with ICD-10 codes
- Secondary conditions
- Risk factors
- Psychosocial considerations

TREATMENT GOALS:
- Short-term objectives (1-3 months)
- Long-term goals (6-12 months)
- Functional outcomes
- Quality of life measures

THERAPEUTIC INTERVENTIONS:
1. Pharmacological treatments
   - Medications with dosing
   - Drug interactions checked
   - Monitoring parameters
2. Non-pharmacological interventions
   - Lifestyle modifications
   - Physical therapy
   - Counseling/education
3. Procedural interventions
   - Surgical considerations
   - Minimally invasive options

MONITORING PLAN:
- Clinical parameters to track
- Laboratory monitoring schedule
- Imaging follow-up
- Functional assessments

CONTINGENCY PLANNING:
- Treatment failure scenarios
- Adverse event management
- Alternative therapies
- Emergency protocols

Include evidence-based guidelines and patient preference considerations.
<end_of_turn>

<start_of_turn>model""",
        best_for=["chronic disease management", "complex cases", "multidisciplinary care"]
    ),
    
    "medication_management": PromptTemplate(
        name="Medication Management Planning",
        description="Comprehensive medication review and optimization",
        template="""<start_of_turn>system
You are a clinical pharmacist specializing in medication management. Focus on safety, efficacy, and adherence optimization.
<end_of_turn>

<start_of_turn>user
Medication Data: {context}

Management Request: {prompt}

Provide medication management plan:

CURRENT MEDICATION REVIEW:
- Current medications with indications
- Dosing appropriateness
- Therapeutic duplications
- Drug-drug interactions
- Drug-disease interactions

MEDICATION OPTIMIZATION:
- Dose adjustments needed
- Therapeutic substitutions
- Discontinued medications with rationale
- New medications recommended

SAFETY CONSIDERATIONS:
- Contraindications identified
- Allergy/intolerance review
- Renal/hepatic dose adjustments
- Age-related considerations
- Pregnancy/lactation safety

MONITORING PLAN:
- Laboratory monitoring requirements
- Clinical monitoring parameters
- Adverse effect surveillance
- Therapeutic drug monitoring

PATIENT EDUCATION:
- Medication counseling points
- Administration instructions
- Adherence strategies
- Warning signs to report

COST CONSIDERATIONS:
- Generic alternatives
- Formulary preferences
- Patient assistance programs
- Insurance coverage issues

Include relevant clinical guidelines and safety alerts.
<end_of_turn>

<start_of_turn>model""",
        best_for=["polypharmacy management", "medication reconciliation", "drug safety"]
    ),
    
    "care_coordination": PromptTemplate(
        name="Care Coordination Planning",
        description="Develops comprehensive care coordination strategies",
        template="""<start_of_turn>system
You are a care coordinator developing comprehensive care plans. Focus on seamless care transitions and multidisciplinary collaboration.
<end_of_turn>

<start_of_turn>user
Patient Care Information: {context}

Coordination Request: {prompt}

Develop care coordination plan:

CARE TEAM COMPOSITION:
- Primary care provider role
- Specialist consultations needed
- Allied health professionals
- Community resources
- Family/caregiver involvement

CARE TRANSITIONS:
- Hospital to home planning
- Specialty care referrals
- Long-term care considerations
- Emergency care protocols

COMMUNICATION PLAN:
- Information sharing protocols
- Regular team meetings
- Patient/family updates
- Documentation standards

SERVICE COORDINATION:
- Appointment scheduling
- Transportation arrangements
- Equipment/supplies needed
- Home health services
- Social services referrals

MONITORING AND FOLLOW-UP:
- Care plan review schedule
- Outcome measurements
- Quality indicators
- Patient satisfaction metrics
- Care gaps identification

PATIENT ENGAGEMENT:
- Self-management education
- Shared decision-making
- Goal setting with patient
- Barrier identification and solutions

RESOURCE OPTIMIZATION:
- Cost-effective care options
- Insurance authorization needs
- Community resource utilization
- Preventive care scheduling

Emphasize patient-centered care and outcome optimization.
<end_of_turn>

<start_of_turn>model""",
        best_for=["complex cases", "chronic disease management", "care transitions"]
    )
}

# Combine all templates
ALL_TEMPLATES = {
    **INTAKE_TEMPLATES,
    **CLINICAL_TEMPLATES,
    **REPORT_TEMPLATES,
    **DIAGNOSTIC_TEMPLATES,
    **TREATMENT_TEMPLATES
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
        "intake": INTAKE_TEMPLATES,
        "clinical": CLINICAL_TEMPLATES,
        "report": REPORT_TEMPLATES,
        "diagnostic": DIAGNOSTIC_TEMPLATES,
        "treatment": TREATMENT_TEMPLATES
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
    """Show how to use medical templates with the Gemma client"""
    example = """
# Using Medical Templates with GemmaClient
from ai.gemma_client import GemmaClient
from rag_functions.templates.prompt_templates import get_template
from rag_functions.ml.vector_operations import select_optimal_templates
from rag_functions.ml.cue_card_extraction import extract_cue_cards

client = GemmaClient()

# Example 1: Automatic Template Selection
document_content = "Patient presents with chest pain, shortness of breath..."
task_description = "Create clinical documentation"

# Automatically select best templates
recommended_templates = select_optimal_templates(document_content, task_description)
print(f"Recommended templates: {recommended_templates}")

# Use the top recommended template
if recommended_templates:
    template_name, confidence = recommended_templates[0]
    template = get_template(template_name)
    
    response = client.generate_response(
        prompt=task_description,
        context=document_content,
        prompt_template=template
    )
    
    # Extract cue cards from LLM output
    cue_cards = extract_cue_cards(response, context_type="clinical")
    formatted_cards = '\\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in cue_cards])
    print(formatted_cards)

# Example 2: Manual Template Selection with Cue Cards
soap_template = get_template("soap_note")
response = client.generate_response(
    prompt="Generate SOAP note for routine follow-up",
    context=encounter_data,
    prompt_template=soap_template
)

# Convert output to medical cue cards
medical_cards = extract_cue_cards(response, context_type="medical")
situational_cards = extract_cue_cards(response, context_type="situational")

print("Medical Context Cards:")
print('\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in medical_cards]))
print("\nSituational Context Cards:")
print('\n'.join([f"{card.context}: {', '.join(card.key_points)}" for card in situational_cards]))

# Example 3: Document Type Analysis
document_scores = analyze_document_type(document_content)
print(f"Document type analysis: {document_scores}")

# Example 4: Complete Workflow
# This would be implemented as:
# result = process_medical_document(
#     content="Patient has diabetes, hypertension, needs medication review",
#     task="Create treatment plan"
# )
"""
    return example

if __name__ == "__main__":
    # Print available medical templates
    print("Medical Documentation Prompt Templates:")
    print("=" * 60)
    
    for category in ["intake", "clinical", "report", "diagnostic", "treatment"]:
        print(f"\n{category.upper()} TEMPLATES:")
        templates = list_templates(category)
        for name, template in templates.items():
            print(f"  - {name}: {template.description}")
            print(f"    Best for: {', '.join(template.best_for)}")
    
    print("\n" + "=" * 60)
    print("Example Usage:")
    print(get_example_usage())