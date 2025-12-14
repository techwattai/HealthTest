import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import guest_booking_prediction_prompt
from Configurations.config import llm_model
from PydanticModels.model import GuestBookingPredictionInput, AIPrediction


def get_guest_booking_prediction(user_input: GuestBookingPredictionInput) -> AIPrediction:
    """
    Analyze guest symptoms during booking to predict urgency level, possible conditions, 
    and recommend appropriate department/specialist.
    
    Args:
        user_input: GuestBookingPredictionInput containing symptoms, description, and optional 
                   patient information
        
    Returns:
        AIPrediction object with urgency level, possible conditions, recommended department, 
        summary, and confidence score
    """
    # Create format instructions for AIPrediction
    format_instructions = """
    You must return a JSON object with:
    - urgency_level: string (one of "Normal", "High", "Emergency")
    - possible_conditions: array of strings (list of possible medical conditions)
    - recommended_department: string (the appropriate medical department/specialty)
    - summary: string (comprehensive clinical summary)
    - confidence_score: float (a value between 0.0 and 1.0 indicating confidence in the assessment)
    
    Example format:
    {
        "urgency_level": "High",
        "possible_conditions": [
            "Congestive Heart Failure",
            "Pulmonary Edema",
            "Acute Coronary Syndrome"
        ],
        "recommended_department": "Cardiology",
        "summary": "Based on symptoms and medical history, patient presents with concerning cardiac symptoms requiring urgent evaluation. Shortness of breath with orthopnea in patient with cardiac history suggests possible heart failure exacerbation.",
        "confidence_score": 0.85
    }
    """
    
    # Format optional fields for the prompt
    age_info = f"Age: {user_input.age}" if user_input.age is not None else ""
    gender_info = f"Gender: {user_input.gender}" if user_input.gender is not None else ""
    
    if user_input.medical_history and len(user_input.medical_history) > 0:
        medical_history_info = f"Medical History: {', '.join(user_input.medical_history)}"
    else:
        medical_history_info = ""
    
    formatted_prompt = guest_booking_prediction_prompt.format(
        symptoms=user_input.symptoms,
        user_description=user_input.user_description,
        age_info=age_info,
        gender_info=gender_info,
        medical_history_info=medical_history_info,
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
        
        # Validate and create AIPrediction object
        if isinstance(data, dict):
            return AIPrediction(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process guest booking prediction response: {str(e)}")

