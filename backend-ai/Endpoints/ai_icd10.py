import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ICD10Agent.icd10_agent import get_icd10_suggestions
from PydanticModels.model import ICD10Input
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-icd10", tags=["AI ICD-10"])
def icd10_endpoint(user_input: ICD10Input):
    """
    Endpoint to suggest appropriate ICD-10 diagnosis codes based on clinical diagnosis text.
    
    Input:
    - diagnosis: Diagnosis description (string)
    
    Output:
    - Array of ICD-10 code suggestion objects with:
      - code: ICD-10 code (string)
      - desc: Code description (string)
      - confidence: Confidence percentage (0-100) (integer)
    """
    try:
        suggestions = get_icd10_suggestions(user_input)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ICD-10 suggestions: {str(e)}")

