# Medical prompt templates
from .prompt_templates import (
    get_template, get_template_info, list_templates, 
    get_template_for_use_case, create_custom_template,
    PromptTemplate, ALL_TEMPLATES
)

__all__ = [
    'get_template',
    'get_template_info', 
    'list_templates',
    'get_template_for_use_case',
    'create_custom_template',
    'PromptTemplate',
    'ALL_TEMPLATES'
]