import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import diagnosis_prompt
from Configurations.config import llm_model
from PydanticModels.model import DiagnosisInput, DiagnosisOutput
from typing import List


def get_diagnosis(user_input: DiagnosisInput) -> List[DiagnosisOutput]:
    """
    Analyze patient symptoms and return possible diagnoses with ICD-10 codes.
    
    Args:
        user_input: DiagnosisInput containing list of symptoms
        
    Returns:
        List of DiagnosisOutput objects with diagnosis, ICD-10 code, and confidence
    """
    # Create format instructions for a list of DiagnosisOutput
    format_instructions = """
    You must return a JSON array of diagnosis objects. Each object should have:
    - diagnosis: string (the diagnosis name)
    - icd10: string (the ICD-10 code)
    - confidence: integer (0-100, percentage confidence)
    
    Example format:
    [
        {
            "diagnosis": "Influenza",
            "icd10": "J11.1",
            "confidence": 87
        },
        {
            "diagnosis": "Common Cold",
            "icd10": "J00",
            "confidence": 72
        }
    ]
    """
    
    formatted_prompt = diagnosis_prompt.format(
        symptoms=user_input.symptoms,
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
            return [DiagnosisOutput(**data)]
        elif isinstance(data, list):
            return [DiagnosisOutput(**item) for item in data]
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process diagnosis response: {str(e)}")

