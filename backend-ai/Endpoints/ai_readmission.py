import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReadmissionAgent.readmission_agent import predict_readmission_risk
from PydanticModels.model import ReadmissionRiskInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-readmission-risk", tags=["AI Readmission Risk"])
def readmission_risk_endpoint(user_input: ReadmissionRiskInput):
    """
    Endpoint to predict likelihood of patient readmission within 30 days of discharge.
    
    This feature helps healthcare providers identify patients at high risk of readmission 
    and implement preventative interventions to reduce avoidable readmissions and improve 
    patient outcomes.
    
    Input:
    - patientId: Patient identifier (string/UUID)
    - demographics: Patient demographics:
      * age: int
      * gender: string
      * insurance: string
      * socialSupport: "none" | "limited" | "moderate" | "strong"
    - clinicalData: Clinical information:
      * primaryDiagnosis: string
      * comorbidities: List[str]
      * lengthOfStay: int (days)
      * previousAdmissions: int
      * emergencyVisits: int
    - discharge: Discharge planning information:
      * medications: int (number of medications)
      * followUpScheduled: bool
      * homeHealthOrdered: bool
      * patientEducationProvided: bool
    
    Output:
    - riskScore: Predicted readmission risk 0-100 (int)
    - riskCategory: "low" | "moderate" | "high" | "very_high"
    - predictedDays: Optional int (when readmission likely to occur)
    - topRiskFactors: Array of risk factors with factor, contribution (0-100), modifiable (bool)
    - preventativeInterventions: Array of interventions with intervention, expectedRiskReduction 
                                  (0-100), cost ("low" | "medium" | "high"), priority (1-10)
    - confidence: Confidence score 0-1 (float)
    """
    try:
        risk_prediction = predict_readmission_risk(user_input)
        return risk_prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing readmission risk prediction: {str(e)}")

