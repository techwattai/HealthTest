import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import drug_interaction_prompt
from Configurations.config import llm_model
from PydanticModels.model import DrugInteractionInput, DrugInteraction
from typing import List


def check_drug_interactions(user_input: DrugInteractionInput) -> List[DrugInteraction]:
    """
    Check for potential drug interactions when multiple medications are prescribed.
    
    Args:
        user_input: DrugInteractionInput containing list of medication names
        
    Returns:
        List of DrugInteraction objects with severity, message, drugs, and recommendation
    """
    # Create format instructions for a list of DrugInteraction
    format_instructions = """
    You must return a JSON array of drug interaction objects. Each object should have:
    - severity: string (one of 'low', 'moderate', 'high', 'severe')
    - msg: string (interaction description)
    - drugs: array of strings (affected drug names)
    - recommendation: string (optional, clinical recommendation)
    
    Example format:
    [
        {
            "severity": "high",
            "msg": "Aspirin + Warfarin: Increased bleeding risk",
            "drugs": ["Aspirin", "Warfarin"],
            "recommendation": "Monitor INR closely. Consider alternative antiplatelet agent."
        }
    ]
    
    If no interactions are found, return an empty array [].
    """
    
    formatted_prompt = drug_interaction_prompt.format(
        drugs=user_input.drugs,
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
            return [DrugInteraction(**data)]
        elif isinstance(data, list):
            # Handle empty list
            if len(data) == 0:
                return []
            return [DrugInteraction(**item) for item in data]
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process drug interaction response: {str(e)}")

