import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import medication_adherence_prompt
from Configurations.config import llm_model
from PydanticModels.model import MedicationAdherenceInput, AdherencePrediction


def predict_medication_adherence(user_input: MedicationAdherenceInput) -> AdherencePrediction:
    """
    Predict patient medication adherence risk based on demographics, prescription complexity, 
    and history.
    
    Args:
        user_input: MedicationAdherenceInput containing patient ID, demographics, prescription, 
                   and history
        
    Returns:
        AdherencePrediction object with adherence probability, risk level, risk factors, 
        and interventions
    """
    # Create format instructions for AdherencePrediction
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "adherenceProbability": 65,
        "riskLevel": "moderate",
        "riskFactors": [
            {
                "factor": "High medication complexity",
                "impact": 25,
                "description": "Complex dosing schedule with 4 doses per day reduces adherence likelihood"
            },
            {
                "factor": "Previous poor adherence",
                "impact": 30,
                "description": "Previous adherence rate of 45% is a strong predictor of future non-adherence"
            }
        ],
        "interventions": [
            {
                "strategy": "Medication reminder system",
                "expectedImprovement": 15,
                "priority": "high"
            },
            {
                "strategy": "Simplify dosing schedule",
                "expectedImprovement": 20,
                "priority": "high"
            },
            {
                "strategy": "Patient education program",
                "expectedImprovement": 10,
                "priority": "medium"
            }
        ]
    }
    
    Important:
    - adherenceProbability is an integer 0-100
    - riskLevel must be one of: "low", "moderate", "high", "very_high"
    - impact for risk factors is an integer 0-100
    - expectedImprovement for interventions is an integer 0-100
    - priority must be one of: "low", "medium", "high"
    - Rank risk factors by impact (highest first)
    - Rank interventions by priority and expected improvement (highest first)
    """
    
    # Format optional fields
    socioeconomic_status = user_input.demographics.socioeconomicStatus or "Not specified"
    education = user_input.demographics.education or "Not specified"
    employment_status = user_input.demographics.employmentStatus or "Not specified"
    cost = f"${user_input.prescription.cost:.2f}" if user_input.prescription.cost is not None else "Not specified"
    previous_adherence = f"{user_input.history.previousAdherenceRate}%" if user_input.history.previousAdherenceRate is not None else "Not available"
    missed_appointments = user_input.history.missedAppointments if user_input.history.missedAppointments is not None else "Not specified"
    has_support = "Yes" if user_input.history.hasSupport else "No" if user_input.history.hasSupport is not None else "Not specified"
    
    formatted_prompt = medication_adherence_prompt.format(
        patient_id=user_input.patientId,
        age=user_input.demographics.age,
        socioeconomic_status=socioeconomic_status,
        education=education,
        employment_status=employment_status,
        medication_count=user_input.prescription.medicationCount,
        doses_per_day=user_input.prescription.dosesPerDay,
        complexity=user_input.prescription.complexity,
        duration=user_input.prescription.duration,
        cost=cost,
        previous_adherence=previous_adherence,
        missed_appointments=missed_appointments,
        has_support=has_support,
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
        
        # Validate and create AdherencePrediction object
        if isinstance(data, dict):
            return AdherencePrediction(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process medication adherence prediction response: {str(e)}")

