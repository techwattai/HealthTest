import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AdherenceAgent.adherence_agent import predict_medication_adherence
from PydanticModels.model import MedicationAdherenceInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-medication-adherence", tags=["AI Medication Adherence"])
def medication_adherence_endpoint(user_input: MedicationAdherenceInput):
    """
    Endpoint to predict patient medication adherence risk based on demographics, 
    prescription complexity, and history.
    
    This feature helps healthcare providers identify patients at risk of medication 
    non-adherence and recommends evidence-based interventions to improve adherence.
    
    Input:
    - patientId: Patient identifier (string/UUID)
    - demographics: Patient demographics:
      * age: int (required)
      * socioeconomicStatus: Optional "low" | "medium" | "high"
      * education: Optional string
      * employmentStatus: Optional string
    - prescription: Prescription details:
      * medicationCount: int (number of medications)
      * dosesPerDay: int
      * complexity: int (1-10 scale)
      * duration: int (days)
      * cost: Optional float
    - history: Adherence history:
      * previousAdherenceRate: Optional float (0-100)
      * missedAppointments: Optional int
      * hasSupport: Optional bool
    
    Output:
    - adherenceProbability: Predicted adherence likelihood 0-100 (int)
    - riskLevel: "low" | "moderate" | "high" | "very_high"
    - riskFactors: Array of risk factors with factor, impact (0-100), description
    - interventions: Array of intervention strategies with strategy, expectedImprovement (0-100), 
                     priority ("low" | "medium" | "high")
    """
    try:
        prediction = predict_medication_adherence(user_input)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing medication adherence prediction: {str(e)}")

