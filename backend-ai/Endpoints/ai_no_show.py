import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NoShowAgent.no_show_agent import predict_no_show
from PydanticModels.model import NoShowPredictionInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-no-show-prediction", tags=["AI No-Show Prediction"])
def no_show_prediction_endpoint(user_input: NoShowPredictionInput):
    """
    Endpoint to predict likelihood of patient missing scheduled appointment.
    
    This feature helps healthcare facilities optimize scheduling, reduce missed appointments, 
    and improve resource utilization by identifying high-risk patients and recommending 
    preventative interventions.
    
    Input:
    - patientId: Patient identifier (string/UUID)
    - appointmentDetails: Appointment information:
      * type: string (appointment type)
      * department: string
      * leadTime: int (days until appointment)
      * dayOfWeek: string
      * timeOfDay: "morning" | "afternoon" | "evening"
    - patientHistory: Historical appointment data:
      * totalAppointments: int
      * missedAppointments: int
      * lastMinuteCancellations: int
      * averageLeadTime: float
    - demographics: Patient demographics:
      * age: int
      * distance: float (miles from facility)
      * transportationAccess: "own" | "public" | "limited"
      * employmentStatus: string
    - engagement: Patient engagement data:
      * remindersSent: int
      * responsesToReminders: int
      * portalActive: bool
    
    Output:
    - probability: Predicted no-show probability 0-100 (int)
    - riskLevel: "low" | "moderate" | "high"
    - contributingFactors: Array of risk factors with factor, weight (0-1 float)
    - recommendations: Array of recommended actions with action, expectedImpact (0-100), 
                       effort ("low" | "medium" | "high")
    """
    try:
        prediction = predict_no_show(user_input)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing no-show prediction: {str(e)}")

