import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import readmission_risk_prompt
from Configurations.config import llm_model
from PydanticModels.model import ReadmissionRiskInput, ReadmissionRisk


def predict_readmission_risk(user_input: ReadmissionRiskInput) -> ReadmissionRisk:
    """
    Predict likelihood of patient readmission within 30 days of discharge.
    
    Args:
        user_input: ReadmissionRiskInput containing patient ID, demographics, clinical data, 
                   and discharge information
        
    Returns:
        ReadmissionRisk object with risk score, category, predicted days, risk factors, 
        interventions, and confidence
    """
    # Create format instructions for ReadmissionRisk
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "riskScore": 65,
        "riskCategory": "high",
        "predictedDays": 10,
        "topRiskFactors": [
            {
                "factor": "Multiple comorbidities",
                "contribution": 25,
                "modifiable": false
            },
            {
                "factor": "No follow-up scheduled",
                "contribution": 20,
                "modifiable": true
            },
            {
                "factor": "Limited social support",
                "contribution": 15,
                "modifiable": false
            }
        ],
        "preventativeInterventions": [
            {
                "intervention": "Schedule follow-up appointment within 7 days",
                "expectedRiskReduction": 25,
                "cost": "low",
                "priority": 10
            },
            {
                "intervention": "Arrange home health services",
                "expectedRiskReduction": 20,
                "cost": "medium",
                "priority": 8
            },
            {
                "intervention": "Medication reconciliation and education",
                "expectedRiskReduction": 15,
                "cost": "low",
                "priority": 9
            }
        ],
        "confidence": 0.82
    }
    
    Important:
    - riskScore is an integer 0-100
    - riskCategory must be one of: "low", "moderate", "high", "very_high"
    - predictedDays is optional (only include if risk is moderate or higher)
    - contribution for risk factors is an integer 0-100 (should sum to approximately 100)
    - expectedRiskReduction for interventions is an integer 0-100
    - cost must be one of: "low", "medium", "high"
    - priority is an integer 1-10 (10 = highest priority)
    - confidence is a float 0-1
    - Rank risk factors by contribution (highest first)
    - Rank interventions by priority (highest first)
    """
    
    # Format clinical data
    comorbidities_str = ", ".join(user_input.clinicalData.comorbidities) if user_input.clinicalData.comorbidities else "None"
    
    formatted_prompt = readmission_risk_prompt.format(
        patient_id=user_input.patientId,
        age=user_input.demographics.age,
        gender=user_input.demographics.gender,
        insurance=user_input.demographics.insurance,
        social_support=user_input.demographics.socialSupport,
        primary_diagnosis=user_input.clinicalData.primaryDiagnosis,
        comorbidities=comorbidities_str,
        length_of_stay=user_input.clinicalData.lengthOfStay,
        previous_admissions=user_input.clinicalData.previousAdmissions,
        emergency_visits=user_input.clinicalData.emergencyVisits,
        medications=user_input.discharge.medications,
        follow_up_scheduled="Yes" if user_input.discharge.followUpScheduled else "No",
        home_health_ordered="Yes" if user_input.discharge.homeHealthOrdered else "No",
        patient_education_provided="Yes" if user_input.discharge.patientEducationProvided else "No",
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
        
        # Validate and create ReadmissionRisk object
        if isinstance(data, dict):
            return ReadmissionRisk(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process readmission risk prediction response: {str(e)}")

