import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BookingAgent.book_agent import get_possible_causes
from PydanticModels.model import UserSymptoms
from fastapi import APIRouter


router = APIRouter()

# Endpoint for possible causes based on user symptoms
@router.post("/ai-appointment",tags=["Appointment Booking Agent"])
def possible_causes_endpoint(user_input: UserSymptoms):
    """
    Endpoint to get possible causes based on user symptoms.
    """
    
    possible_causes = get_possible_causes(user_input)
    return possible_causes


