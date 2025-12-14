import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LabInterpretationAgent.lab_interpretation_agent import interpret_lab_results
from PydanticModels.model import LabInterpretationInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-lab-interpretation", tags=["AI Lab Interpretation"])
def lab_interpretation_endpoint(user_input: LabInterpretationInput):
    """
    Endpoint for AI-assisted interpretation of lab results in clinical context.
    
    This feature helps healthcare providers interpret lab results by analyzing values 
    against reference ranges, considering patient symptoms, diagnoses, medications, 
    and demographics to provide clinical insights and recommendations.
    
    Input:
    - patientId: Patient identifier (string/UUID)
    - labResults: Array of lab test results, each with:
      * testName: string
      * value: float
      * unit: string
      * referenceRange: {min: float, max: float}
    - clinicalContext: Patient clinical information:
      * symptoms: List[str]
      * currentDiagnoses: List[str]
      * medications: List[str]
      * age: int
      * gender: string
    
    Output:
    - summary: Clinical summary of lab findings (string)
    - abnormalFindings: Array of abnormal findings with:
      * test: string
      * significance: "critical" | "high" | "moderate" | "low"
      * clinicalImplications: List[str]
      * possibleCauses: List[str]
      * recommendedActions: List[str]
    - suggestedFollowUp: Array of recommended follow-up tests with:
      * test: string
      * reason: string
      * urgency: "immediate" | "within_24h" | "within_week" | "routine"
    - confidence: Confidence score 0-1 (float)
    """
    try:
        interpretation = interpret_lab_results(user_input)
        return interpretation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing lab result interpretation: {str(e)}")

