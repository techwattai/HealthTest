import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HealthAnalysisAgent.health_analysis_agent import get_comprehensive_health_analysis
from PydanticModels.model import HealthAnalysisInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-health-analysis", tags=["AI Health Analysis"])
def comprehensive_health_analysis_endpoint(user_input: HealthAnalysisInput):
    """
    Endpoint for comprehensive AI-powered health analysis combining symptoms, vitals, 
    and medical history to provide:
    - Possible conditions with probability and severity
    - Recommended doctors/specialists with match scores
    - Risk factors
    - Care recommendations (remedies)
    - Follow-up plans
    
    Input:
    - age: Patient age (string)
    - gender: Patient gender (string)
    - symptoms: Comma-separated symptoms (string)
    - vitals: Object with bloodPressure, heartRate, temperature, oxygenSat (strings)
    - medicalHistory: Optional list of known conditions (List[str])
    
    Output:
    - conditions: Array of condition objects with name, probability, severity, description
    - recommendedDoctors: Array of doctor objects with name, specialty, match, availability, 
                          experience, rating
    - remedies: List of care recommendations (List[str])
    - urgency: "routine" | "urgent" | "emergency"
    - confidence: Overall confidence score 0-100 (int)
    - riskFactors: List of identified risk factors (List[str])
    - followUpRecommendations: List of follow-up actions (List[str])
    """
    try:
        analysis = get_comprehensive_health_analysis(user_input)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing comprehensive health analysis: {str(e)}")

