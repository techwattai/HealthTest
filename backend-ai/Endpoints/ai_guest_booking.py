import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GuestBookingAgent.guest_booking_agent import get_guest_booking_prediction
from PydanticModels.model import GuestBookingPredictionInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-guest-booking", tags=["AI Guest Booking"])
def guest_booking_prediction_endpoint(user_input: GuestBookingPredictionInput):
    """
    Endpoint to analyze guest symptoms during booking to predict urgency level, possible 
    conditions, and recommend appropriate department/specialist.
    
    Input:
    - symptoms: Array of tagged symptoms (List[str])
    - user_description: Free-text symptom description (string)
    - age: Patient age (optional int)
    - gender: Patient gender (optional string)
    - medical_history: Known conditions (optional List[str])
    
    Output:
    - urgency_level: "Normal" | "High" | "Emergency"
    - possible_conditions: List of possible medical conditions (List[str])
    - recommended_department: Appropriate medical department (string)
    - summary: Comprehensive clinical summary (string)
    - confidence_score: Confidence score 0-1 (float)
    """
    try:
        prediction = get_guest_booking_prediction(user_input)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing guest booking prediction: {str(e)}")

