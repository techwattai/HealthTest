import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import no_show_prediction_prompt
from Configurations.config import llm_model
from PydanticModels.model import NoShowPredictionInput, NoShowPrediction


def predict_no_show(user_input: NoShowPredictionInput) -> NoShowPrediction:
    """
    Predict likelihood of patient missing scheduled appointment.
    
    Args:
        user_input: NoShowPredictionInput containing patient ID, appointment details, 
                   patient history, demographics, and engagement data
        
    Returns:
        NoShowPrediction object with probability, risk level, contributing factors, 
        and recommendations
    """
    # Create format instructions for NoShowPrediction
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "probability": 45,
        "riskLevel": "moderate",
        "contributingFactors": [
            {
                "factor": "High historical no-show rate (30%)",
                "weight": 0.35
            },
            {
                "factor": "Long lead time (45 days)",
                "weight": 0.25
            },
            {
                "factor": "Limited transportation access",
                "weight": 0.20
            },
            {
                "factor": "Low engagement with reminders",
                "weight": 0.20
            }
        ],
        "recommendations": [
            {
                "action": "Send reminder 24 hours before appointment",
                "expectedImpact": 20,
                "effort": "low"
            },
            {
                "action": "Offer telehealth option",
                "expectedImpact": 25,
                "effort": "medium"
            },
            {
                "action": "Reschedule to shorter lead time if possible",
                "expectedImpact": 15,
                "effort": "low"
            }
        ]
    }
    
    Important:
    - probability is an integer 0-100
    - riskLevel must be one of: "low", "moderate", "high"
    - weight for contributing factors is a float 0-1 (should sum to approximately 1.0)
    - expectedImpact for recommendations is an integer 0-100
    - effort must be one of: "low", "medium", "high"
    - Rank contributing factors by weight (highest first)
    - Rank recommendations by expected impact (highest first)
    """
    
    # Calculate historical no-show rate
    historical_no_show_rate = 0
    if user_input.patientHistory.totalAppointments > 0:
        historical_no_show_rate = (user_input.patientHistory.missedAppointments / 
                                   user_input.patientHistory.totalAppointments) * 100
    
    formatted_prompt = no_show_prediction_prompt.format(
        patient_id=user_input.patientId,
        appointment_type=user_input.appointmentDetails.type,
        department=user_input.appointmentDetails.department,
        lead_time=user_input.appointmentDetails.leadTime,
        day_of_week=user_input.appointmentDetails.dayOfWeek,
        time_of_day=user_input.appointmentDetails.timeOfDay,
        total_appointments=user_input.patientHistory.totalAppointments,
        missed_appointments=user_input.patientHistory.missedAppointments,
        last_minute_cancellations=user_input.patientHistory.lastMinuteCancellations,
        average_lead_time=user_input.patientHistory.averageLeadTime,
        age=user_input.demographics.age,
        distance=user_input.demographics.distance,
        transportation_access=user_input.demographics.transportationAccess,
        employment_status=user_input.demographics.employmentStatus,
        reminders_sent=user_input.engagement.remindersSent,
        responses_to_reminders=user_input.engagement.responsesToReminders,
        portal_active="Yes" if user_input.engagement.portalActive else "No",
        format_instructions=format_instructions
    )

    response = llm_model.LLM().invoke(formatted_prompt)
    
    # Parse the JSON response
    try:
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        # Validate and create NoShowPrediction object
        if isinstance(data, dict):
            return NoShowPrediction(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process no-show prediction response: {str(e)}")

