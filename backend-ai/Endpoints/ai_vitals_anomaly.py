import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VitalsAnomalyAgent.vitals_anomaly_agent import detect_vitals_anomalies
from PydanticModels.model import VitalsAnomalyInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-vitals-anomaly", tags=["AI Vitals Anomaly Detection"])
def vitals_anomaly_detection_endpoint(user_input: VitalsAnomalyInput):
    """
    Endpoint for real-time monitoring of patient vital signs to detect anomalies and trigger alerts.
    
    This is a CRITICAL patient safety feature that analyzes vital signs in real-time and provides:
    - Anomaly detection with severity assessment
    - Individual vital sign analysis with deviation scores
    - Clinical recommendations
    - Alert level determination (none, monitor, notify_nurse, notify_doctor, emergency)
    
    Input:
    - patientId: Patient identifier (string/UUID)
    - timestamp: ISO date string of measurement time
    - vitals: Object with optional vital signs:
      * heartRate: float (bpm)
      * bloodPressure: {systolic: float, diastolic: float} (mmHg)
      * temperature: float (Celsius)
      * oxygenSaturation: float (percentage)
      * respiratoryRate: float (per minute)
    - patientContext: Patient information:
      * age: int
      * conditions: List[str]
      * medications: List[str]
      * baseline: Optional baseline vitals for this patient
    
    Output:
    - isAnomaly: Boolean indicating if anomalies detected
    - severity: "low" | "medium" | "high" | "critical"
    - anomalies: Array of anomaly objects with vitalSign, currentValue, expectedRange, 
                 deviationScore, trendDirection
    - recommendations: List of clinical actions (List[str])
    - alertLevel: "none" | "monitor" | "notify_nurse" | "notify_doctor" | "emergency"
    - confidence: Confidence score 0-1 (float)
    """
    try:
        detection = detect_vitals_anomalies(user_input)
        return detection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing vital signs anomaly detection: {str(e)}")

