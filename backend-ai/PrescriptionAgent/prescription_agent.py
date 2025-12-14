import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import prescription_support_prompt
from Configurations.config import llm_model
from PydanticModels.model import PrescriptionSupportInput, PrescriptionRecommendation


def get_prescription_recommendations(user_input: PrescriptionSupportInput) -> PrescriptionRecommendation:
    """
    AI-powered recommendations for optimal medication selection based on diagnosis, 
    patient factors, and evidence-based guidelines.
    
    Args:
        user_input: PrescriptionSupportInput containing diagnosis, patient factors, 
                   and optional preferences
        
    Returns:
        PrescriptionRecommendation object with primary recommendations, alternatives, 
        contraindications, warnings, and drug interactions
    """
    # Create format instructions for PrescriptionRecommendation
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "primaryRecommendations": [
            {
                "medication": "Amoxicillin",
                "dose": "500 mg",
                "frequency": "three times daily",
                "duration": "7-10 days",
                "route": "oral",
                "rationale": "First-line treatment for community-acquired pneumonia per IDSA guidelines",
                "evidenceLevel": "A",
                "cost": "low",
                "sideEffects": [
                    "Diarrhea",
                    "Nausea",
                    "Rash"
                ],
                "monitoring": [
                    "Monitor for allergic reactions",
                    "Assess response to treatment"
                ]
            }
        ],
        "alternatives": [
            {
                "medication": "Azithromycin",
                "whenToConsider": "If patient has penicillin allergy",
                "advantages": [
                    "Once daily dosing",
                    "Shorter duration (5 days)"
                ],
                "disadvantages": [
                    "Higher cost",
                    "QT prolongation risk"
                ]
            }
        ],
        "contraindications": [
            "Penicillin (if allergic)",
            "Fluoroquinolones (if <18 years)"
        ],
        "warnings": [
            "Adjust dose for renal impairment",
            "Monitor liver function with prolonged use"
        ],
        "drugInteractions": [
            {
                "interaction": "Amoxicillin may reduce effectiveness of oral contraceptives",
                "severity": "moderate",
                "management": "Advise additional contraceptive method during treatment"
            }
        ]
    }
    
    Important:
    - evidenceLevel must be one of: "A", "B", "C"
    - cost must be one of: "low", "medium", "high"
    - severity for drug interactions must be one of: "low", "moderate", "high"
    - Rank primary recommendations by evidence level and appropriateness (best first)
    - Rank alternatives by when to consider (most common scenarios first)
    - Rank drug interactions by severity (most severe first)
    """
    
    # Format optional patient factors
    weight_info = f"Weight: {user_input.patientFactors.weight} kg" if user_input.patientFactors.weight is not None else ""
    
    if user_input.patientFactors.kidneyFunction:
        kidney_function_info = (
            f"Kidney Function: Creatinine {user_input.patientFactors.kidneyFunction.creatinine} mg/dL, "
            f"GFR {user_input.patientFactors.kidneyFunction.gfr} mL/min/1.73m²"
        )
    else:
        kidney_function_info = "Kidney Function: Not specified"
    
    liver_function_info = f"Liver Function: {user_input.patientFactors.liverFunction}" if user_input.patientFactors.liverFunction else "Liver Function: Not specified"
    
    pregnancy_info = "Pregnancy: Yes" if user_input.patientFactors.pregnancy else "Pregnancy: No" if user_input.patientFactors.pregnancy is not None else "Pregnancy: Not specified"
    
    # Format lists
    allergies_str = ", ".join(user_input.patientFactors.allergies) if user_input.patientFactors.allergies else "None"
    current_medications_str = ", ".join(user_input.patientFactors.currentMedications) if user_input.patientFactors.currentMedications else "None"
    comorbidities_str = ", ".join(user_input.patientFactors.comorbidities) if user_input.patientFactors.comorbidities else "None"
    
    # Format preferences
    preferences_parts = []
    if user_input.preferences:
        if user_input.preferences.costSensitive is not None:
            preferences_parts.append(f"Cost Sensitive: {'Yes' if user_input.preferences.costSensitive else 'No'}")
        if user_input.preferences.preferGeneric is not None:
            preferences_parts.append(f"Prefer Generic: {'Yes' if user_input.preferences.preferGeneric else 'No'}")
        if user_input.preferences.routePreference:
            preferences_parts.append(f"Route Preference: {user_input.preferences.routePreference}")
    
    preferences_info = "\n".join(preferences_parts) if preferences_parts else "No specific preferences"
    
    formatted_prompt = prescription_support_prompt.format(
        diagnosis=user_input.diagnosis,
        age=user_input.patientFactors.age,
        weight_info=weight_info,
        kidney_function_info=kidney_function_info,
        liver_function_info=liver_function_info,
        allergies=allergies_str,
        current_medications=current_medications_str,
        comorbidities=comorbidities_str,
        pregnancy_info=pregnancy_info,
        preferences_info=preferences_info,
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
        
        # Validate and create PrescriptionRecommendation object
        if isinstance(data, dict):
            return PrescriptionRecommendation(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process prescription recommendation response: {str(e)}")

