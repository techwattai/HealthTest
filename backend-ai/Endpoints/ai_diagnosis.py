import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DiagnosisAgent.diagnosis_agent import get_diagnosis
from PydanticModels.model import DiagnosisInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-diagnosis", tags=["AI Diagnosis"])
def diagnosis_endpoint(user_input: DiagnosisInput):
    """
    Endpoint to analyze patient symptoms and provide possible diagnosis suggestions with ICD-10 codes.
    
    Input:
    - symptoms: Array of symptom descriptions
    
    Output:
    - Array of diagnosis objects with:
      - diagnosis: Primary diagnosis name
      - icd10: ICD-10 code
      - confidence: Percentage (0-100)
    """
    try:
        diagnoses = get_diagnosis(user_input)
        return diagnoses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing diagnosis: {str(e)}")

