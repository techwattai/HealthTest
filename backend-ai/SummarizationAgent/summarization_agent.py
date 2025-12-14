import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import notes_summarization_prompt
from Configurations.config import llm_model
from PydanticModels.model import NotesSummarizationInput, SummarizedNotes


def summarize_notes(user_input: NotesSummarizationInput) -> SummarizedNotes:
    """
    Transform raw clinical notes into structured, formatted medical documentation.
    
    Args:
        user_input: NotesSummarizationInput containing raw clinical notes
        
    Returns:
        SummarizedNotes object with structured summary and confidence score
    """
    # Create format instructions for SummarizedNotes
    format_instructions = """
    You must return a JSON object with:
    - summary: string (the structured medical summary in standard medical format)
    - confidence: float (a value between 0.0 and 1.0 indicating confidence in the summarization)
    
    Example format:
    {
        "summary": "Chief Complaint: Chest pain radiating to left arm\\n\\nHPI: Patient reports acute chest pain onset 2 hours prior to presentation. Pain radiates to left arm. Positive history of hypertension.\\n\\nVitals: BP 165/98, HR 92\\n\\nAssessment: Rule out acute coronary syndrome\\n\\nPlan: EKG, Troponin levels, Cardiology consult",
        "confidence": 0.91
    }
    """
    
    formatted_prompt = notes_summarization_prompt.format(
        notes=user_input.notes,
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
        
        # Validate and create SummarizedNotes object
        if isinstance(data, dict):
            return SummarizedNotes(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process summarization response: {str(e)}")

