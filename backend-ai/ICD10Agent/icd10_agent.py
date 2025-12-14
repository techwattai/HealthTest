import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import icd10_suggestion_prompt
from Configurations.config import llm_model
from PydanticModels.model import ICD10Input, ICD10Suggestion
from typing import List


def get_icd10_suggestions(user_input: ICD10Input) -> List[ICD10Suggestion]:
    """
    Suggest appropriate ICD-10 diagnosis codes based on clinical diagnosis text.
    
    Args:
        user_input: ICD10Input containing diagnosis description
        
    Returns:
        List of ICD10Suggestion objects with code, description, and confidence
    """
    # Create format instructions for a list of ICD10Suggestion
    format_instructions = """
    You must return a JSON array of ICD-10 code suggestion objects. Each object should have:
    - code: string (the ICD-10 code)
    - desc: string (the official ICD-10 code description)
    - confidence: integer (0-100, percentage confidence)
    
    Example format:
    [
        {
            "code": "E11.40",
            "desc": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
            "confidence": 95
        },
        {
            "code": "E11.9",
            "desc": "Type 2 diabetes mellitus without complications",
            "confidence": 78
        }
    ]
    """
    
    formatted_prompt = icd10_suggestion_prompt.format(
        diagnosis=user_input.diagnosis,
        format_instructions=format_instructions
    )

    response = llm_model.LLM().invoke(formatted_prompt)
    
    # Parse the JSON response
    try:
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            return [ICD10Suggestion(**data)]
        elif isinstance(data, list):
            return [ICD10Suggestion(**item) for item in data]
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process ICD-10 suggestion response: {str(e)}")

