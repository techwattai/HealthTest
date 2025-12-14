import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PrescriptionAgent.prescription_agent import get_prescription_recommendations
from PydanticModels.model import PrescriptionSupportInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-prescription-support", tags=["AI Prescription Support"])
def prescription_support_endpoint(user_input: PrescriptionSupportInput):
    """
    Endpoint for AI-powered recommendations for optimal medication selection based on 
    diagnosis, patient factors, and evidence-based guidelines.
    
    This feature helps healthcare providers make evidence-based medication decisions by 
    considering patient-specific factors, allergies, comorbidities, drug interactions, 
    and clinical guidelines.
    
    Input:
    - diagnosis: Primary diagnosis (string)
    - patientFactors: Patient-specific factors:
      * age: int (required)
      * weight: Optional float (kg)
      * kidneyFunction: Optional {creatinine: float, gfr: float}
      * liverFunction: Optional string
      * allergies: List[str] (required)
      * currentMedications: List[str] (required)
      * comorbidities: List[str] (required)
      * pregnancy: Optional bool
    - preferences: Optional prescription preferences:
      * costSensitive: Optional bool
      * preferGeneric: Optional bool
      * routePreference: Optional "oral" | "iv" | "im" | "any"
    
    Output:
    - primaryRecommendations: Array of recommended medications with:
      * medication, dose, frequency, duration, route
      * rationale, evidenceLevel ("A" | "B" | "C")
      * cost ("low" | "medium" | "high")
      * sideEffects, monitoring (both List[str])
    - alternatives: Array of alternative medications with:
      * medication, whenToConsider, advantages, disadvantages
    - contraindications: List[str]
    - warnings: List[str]
    - drugInteractions: Array of interactions with:
      * interaction, severity ("low" | "moderate" | "high"), management
    """
    try:
        recommendations = get_prescription_recommendations(user_input)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prescription recommendations: {str(e)}")

