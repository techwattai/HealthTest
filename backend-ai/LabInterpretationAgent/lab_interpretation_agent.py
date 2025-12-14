import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import lab_interpretation_prompt
from Configurations.config import llm_model
from PydanticModels.model import LabInterpretationInput, LabInterpretation


def interpret_lab_results(user_input: LabInterpretationInput) -> LabInterpretation:
    """
    AI-assisted interpretation of lab results in clinical context.
    
    Args:
        user_input: LabInterpretationInput containing patient ID, lab results, and 
                   clinical context
        
    Returns:
        LabInterpretation object with summary, abnormal findings, suggested follow-up, 
        and confidence
    """
    # Create format instructions for LabInterpretation
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "summary": "Comprehensive clinical summary of lab findings...",
        "abnormalFindings": [
            {
                "test": "Hemoglobin",
                "significance": "high",
                "clinicalImplications": [
                    "Significant anemia may indicate blood loss or impaired production",
                    "Correlates with patient's reported fatigue"
                ],
                "possibleCauses": [
                    "Iron deficiency",
                    "Chronic blood loss",
                    "Medication-induced anemia"
                ],
                "recommendedActions": [
                    "Check iron studies (ferritin, TIBC)",
                    "Consider GI workup for occult bleeding",
                    "Review medications for potential causes"
                ]
            }
        ],
        "suggestedFollowUp": [
            {
                "test": "Iron studies (Ferritin, TIBC, Iron)",
                "reason": "To determine cause of anemia",
                "urgency": "within_week"
            },
            {
                "test": "Fecal occult blood test",
                "reason": "Rule out GI bleeding as cause of anemia",
                "urgency": "within_24h"
            }
        ],
        "confidence": 0.87
    }
    
    Important:
    - significance must be one of: "critical", "high", "moderate", "low"
    - urgency must be one of: "immediate", "within_24h", "within_week", "routine"
    - confidence is a float 0-1
    - Only include tests that are actually abnormal in abnormalFindings
    - Rank abnormal findings by significance (most critical first)
    - Rank suggested follow-up by urgency (most urgent first)
    """
    
    # Format lab results information
    lab_results_parts = []
    for lab in user_input.labResults:
        is_abnormal = lab.value < lab.referenceRange.min or lab.value > lab.referenceRange.max
        status = "ABNORMAL" if is_abnormal else "Normal"
        lab_results_parts.append(
            f"  - {lab.testName}: {lab.value} {lab.unit} "
            f"(Reference: {lab.referenceRange.min}-{lab.referenceRange.max} {lab.unit}) [{status}]"
        )
    
    lab_results_info = "\n".join(lab_results_parts)
    
    # Format clinical context
    symptoms_str = ", ".join(user_input.clinicalContext.symptoms) if user_input.clinicalContext.symptoms else "None reported"
    diagnoses_str = ", ".join(user_input.clinicalContext.currentDiagnoses) if user_input.clinicalContext.currentDiagnoses else "None"
    medications_str = ", ".join(user_input.clinicalContext.medications) if user_input.clinicalContext.medications else "None"
    
    formatted_prompt = lab_interpretation_prompt.format(
        patient_id=user_input.patientId,
        lab_results_info=lab_results_info,
        age=user_input.clinicalContext.age,
        gender=user_input.clinicalContext.gender,
        symptoms=symptoms_str,
        diagnoses=diagnoses_str,
        medications=medications_str,
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
        
        # Validate and create LabInterpretation object
        if isinstance(data, dict):
            return LabInterpretation(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process lab result interpretation response: {str(e)}")

