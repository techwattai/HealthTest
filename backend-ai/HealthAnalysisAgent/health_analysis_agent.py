import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import comprehensive_health_analysis_prompt
from Configurations.config import llm_model
from PydanticModels.model import HealthAnalysisInput, ComprehensiveHealthAnalysis


def get_comprehensive_health_analysis(user_input: HealthAnalysisInput) -> ComprehensiveHealthAnalysis:
    """
    Comprehensive AI-powered analysis combining symptoms, vitals, and medical history 
    to provide possible conditions, recommended doctors, risk factors, care recommendations, 
    and follow-up plans.
    
    Args:
        user_input: HealthAnalysisInput containing age, gender, symptoms, vitals, and 
                   optional medical history
        
    Returns:
        ComprehensiveHealthAnalysis object with conditions, recommended doctors, remedies, 
        urgency, confidence, risk factors, and follow-up recommendations
    """
    # Create format instructions for ComprehensiveHealthAnalysis
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "conditions": [
            {
                "name": "Condition name",
                "probability": 87,
                "severity": "severe",
                "description": "Clinical description"
            }
        ],
        "recommendedDoctors": [
            {
                "name": "Dr. Sarah Johnson",
                "specialty": "Cardiology",
                "match": 94,
                "availability": "Available today",
                "experience": "15 years",
                "rating": 4.8
            }
        ],
        "remedies": [
            "Immediate blood pressure control",
            "Neurological evaluation"
        ],
        "urgency": "urgent",
        "confidence": 89,
        "riskFactors": [
            "Uncontrolled hypertension",
            "Diabetes mellitus"
        ],
        "followUpRecommendations": [
            "Emergency department evaluation within 2 hours",
            "Blood pressure monitoring every 30 minutes"
        ]
    }
    
    Important:
    - urgency must be one of: "routine", "urgent", "emergency"
    - severity for conditions must be one of: "mild", "moderate", "severe"
    - probability and confidence are integers 0-100
    - rating is a float 0-5
    - match is an integer 0-100
    """
    
    # Format medical history if provided
    if user_input.medicalHistory and len(user_input.medicalHistory) > 0:
        medical_history_info = f"Medical History: {', '.join(user_input.medicalHistory)}"
    else:
        medical_history_info = "Medical History: None provided"
    
    formatted_prompt = comprehensive_health_analysis_prompt.format(
        age=user_input.age,
        gender=user_input.gender,
        symptoms=user_input.symptoms,
        blood_pressure=user_input.vitals.bloodPressure,
        heart_rate=user_input.vitals.heartRate,
        temperature=user_input.vitals.temperature,
        oxygen_sat=user_input.vitals.oxygenSat,
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
        
        # Validate and create ComprehensiveHealthAnalysis object
        if isinstance(data, dict):
            return ComprehensiveHealthAnalysis(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process comprehensive health analysis response: {str(e)}")

