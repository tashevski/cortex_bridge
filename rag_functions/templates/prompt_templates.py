from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    description: str
    template: str
    best_for: List[str]

# Essential medical templates
ALL_TEMPLATES = {
    "soap_note": PromptTemplate(
        name="SOAP Note",
        description="Creates SOAP notes",
        template="""<start_of_turn>system
You are a medical documentation specialist.
<end_of_turn>

<start_of_turn>user
Patient Data: {context}
Task: {prompt}

Create a SOAP note with:
SUBJECTIVE: Chief complaint, history
OBJECTIVE: Vital signs, examination
ASSESSMENT: Clinical impression
PLAN: Treatment recommendations
<end_of_turn>

<start_of_turn>model""",
        best_for=["clinical documentation", "patient visits"]
    ),
    
    "diagnostic_report": PromptTemplate(
        name="Diagnostic Report",
        description="Creates diagnostic reports",
        template="""<start_of_turn>system
You are a medical diagnostician.
<end_of_turn>

<start_of_turn>user
Clinical Data: {context}
Request: {prompt}

Generate diagnostic report with:
PRESENTATION: Key symptoms and findings
ANALYSIS: Test results and correlation
IMPRESSION: Primary diagnosis
RECOMMENDATIONS: Next steps
<end_of_turn>

<start_of_turn>model""",
        best_for=["diagnostic workups", "test interpretation"]
    ),
    
    "treatment_plan": PromptTemplate(
        name="Treatment Plan",
        description="Creates treatment plans",
        template="""<start_of_turn>system
You are a healthcare provider creating treatment plans.
<end_of_turn>

<start_of_turn>user
Patient Data: {context}
Planning Request: {prompt}

Develop treatment plan with:
PROBLEMS: Primary diagnoses
GOALS: Treatment objectives
INTERVENTIONS: Medications and procedures
MONITORING: Follow-up schedule
<end_of_turn>

<start_of_turn>model""",
        best_for=["treatment planning", "care coordination"]
    )
}

def get_template(name: str) -> str:
    template_obj = ALL_TEMPLATES.get(name)
    return template_obj.template if template_obj else None

def list_templates() -> Dict[str, PromptTemplate]:
    return ALL_TEMPLATES