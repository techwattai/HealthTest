import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import vitals_anomaly_detection_prompt
from Configurations.config import llm_model
from PydanticModels.model import VitalsAnomalyInput, VitalsAnomalyDetection


def detect_vitals_anomalies(user_input: VitalsAnomalyInput) -> VitalsAnomalyDetection:
    """
    Real-time monitoring of patient vital signs to detect anomalies and trigger alerts.
    
    Args:
        user_input: VitalsAnomalyInput containing patient ID, timestamp, vitals, and 
                   patient context
        
    Returns:
        VitalsAnomalyDetection object with anomaly status, severity, anomalies, 
        recommendations, alert level, and confidence
    """
    # Create format instructions for VitalsAnomalyDetection
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "isAnomaly": true,
        "severity": "high",
        "anomalies": [
            {
                "vitalSign": "oxygenSaturation",
                "currentValue": 88,
                "expectedRange": {
                    "min": 95,
                    "max": 100
                },
                "deviationScore": 0.87,
                "trendDirection": "worsening"
            },
            {
                "vitalSign": "bloodPressureSystolic",
                "currentValue": 165,
                "expectedRange": {
                    "min": 120,
                    "max": 130
                },
                "deviationScore": 0.70,
                "trendDirection": "worsening"
            },
            {
                "vitalSign": "bloodPressureDiastolic",
                "currentValue": 105,
                "expectedRange": {
                    "min": 80,
                    "max": 85
                },
                "deviationScore": 0.80,
                "trendDirection": "worsening"
            }
        ],
        "recommendations": [
            "Administer supplemental oxygen",
            "Check airway patency",
            "Consider arterial blood gas analysis"
        ],
        "alertLevel": "notify_doctor",
        "confidence": 0.94
    }
    
    CRITICAL FORMATTING RULES:
    - currentValue must be a NUMBER (float or int), NEVER a string like "165/105"
    - For blood pressure, create TWO separate anomalies:
      * One with vitalSign: "bloodPressureSystolic" and currentValue as the systolic number (e.g., 165)
      * One with vitalSign: "bloodPressureDiastolic" and currentValue as the diastolic number (e.g., 105)
    - expectedRange.min and expectedRange.max must be NUMBERS, not strings
    - severity must be one of: "low", "medium", "high", "critical"
    - alertLevel must be one of: "none", "monitor", "notify_nurse", "notify_doctor", "emergency"
    - trendDirection must be one of: "stable", "improving", "worsening"
    - deviationScore is a float 0-1
    - confidence is a float 0-1
    - If no anomalies detected, isAnomaly should be false and anomalies array should be empty
    """
    
    # Format vitals information
    vitals_parts = []
    if user_input.vitals.heartRate is not None:
        vitals_parts.append(f"  - Heart Rate: {user_input.vitals.heartRate} bpm")
    if user_input.vitals.bloodPressure is not None:
        vitals_parts.append(f"  - Blood Pressure: {user_input.vitals.bloodPressure.systolic}/{user_input.vitals.bloodPressure.diastolic} mmHg")
    if user_input.vitals.temperature is not None:
        vitals_parts.append(f"  - Temperature: {user_input.vitals.temperature}°C")
    if user_input.vitals.oxygenSaturation is not None:
        vitals_parts.append(f"  - Oxygen Saturation: {user_input.vitals.oxygenSaturation}%")
    if user_input.vitals.respiratoryRate is not None:
        vitals_parts.append(f"  - Respiratory Rate: {user_input.vitals.respiratoryRate} /min")
    
    vitals_info = "\n".join(vitals_parts) if vitals_parts else "  - No vital signs provided"
    
    # Format conditions and medications
    conditions_str = ", ".join(user_input.patientContext.conditions) if user_input.patientContext.conditions else "None"
    medications_str = ", ".join(user_input.patientContext.medications) if user_input.patientContext.medications else "None"
    
    # Format baseline information
    baseline_parts = []
    if user_input.patientContext.baseline:
        if user_input.patientContext.baseline.heartRate:
            baseline_parts.append(f"  - Heart Rate Baseline: {user_input.patientContext.baseline.heartRate.min}-{user_input.patientContext.baseline.heartRate.max} bpm")
        if user_input.patientContext.baseline.bloodPressure:
            baseline_parts.append(f"  - Blood Pressure Baseline: {user_input.patientContext.baseline.bloodPressure.systolic}/{user_input.patientContext.baseline.bloodPressure.diastolic} mmHg")
    
    baseline_info = "\n".join(baseline_parts) if baseline_parts else "  - No baseline data available (using standard ranges)"
    
    formatted_prompt = vitals_anomaly_detection_prompt.format(
        patient_id=user_input.patientId,
        timestamp=user_input.timestamp,
        vitals_info=vitals_info,
        age=user_input.patientContext.age,
        conditions=conditions_str,
        medications=medications_str,
        baseline_info=baseline_info,
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
        
        # Validate and create VitalsAnomalyDetection object
        if isinstance(data, dict):
            return VitalsAnomalyDetection(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process vital signs anomaly detection response: {str(e)}")

