import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Prompts.prompt import imaging_analysis_prompt
from Configurations.config import llm_model
from PydanticModels.model import ImagingAnalysisInput, ImagingAnalysis


def analyze_medical_imaging(user_input: ImagingAnalysisInput) -> ImagingAnalysis:
    """
    AI-assisted analysis of medical images (X-rays, CT, MRI, Ultrasound).
    
    NOTE: This is a future enhancement that requires:
    - Vision-capable AI model (GPT-4 Vision, specialized medical imaging AI)
    - Direct image processing capabilities
    - Regulatory approval for medical imaging AI
    - Integration with PACS systems
    
    Current implementation provides structure for future vision model integration.
    
    Args:
        user_input: ImagingAnalysisInput containing image type, URL, body part, 
                   clinical indication, patient demographics, and optional prior findings
        
    Returns:
        ImagingAnalysis object with findings, impression, recommendations, comparison, 
        critical findings flag, and radiologist review requirement
    """
    # Create format instructions for ImagingAnalysis
    format_instructions = """
    You must return a JSON object with the following structure:
    {
        "findings": [
            {
                "location": "Right lower lobe",
                "description": "Patchy opacities consistent with pneumonia",
                "severity": "moderate",
                "confidence": 0.85,
                "coordinates": {
                    "x": 0.3,
                    "y": 0.6,
                    "width": 0.2,
                    "height": 0.15
                }
            },
            {
                "location": "Heart",
                "description": "Cardiomediastinal silhouette within normal limits",
                "severity": "normal",
                "confidence": 0.95
            }
        ],
        "impression": "Patchy opacities in the right lower lobe consistent with pneumonia. 
                      No pneumothorax or pleural effusion. Heart size normal.",
        "recommendations": [
            "Clinical correlation recommended",
            "Consider follow-up chest X-ray in 7-10 days to assess resolution",
            "Antibiotic therapy as clinically indicated"
        ],
        "comparison": "No prior imaging available for comparison",
        "criticalFindings": false,
        "radiologistReviewRequired": true
    }
    
    Important:
    - severity must be one of: "normal", "mild", "moderate", "severe"
    - confidence is a float 0-1
    - coordinates are optional and should be normalized 0-1 (only include if location can be specified)
    - criticalFindings is true if any finding requires immediate attention
    - radiologistReviewRequired is true for any abnormal findings or if confidence is low
    - Rank findings by severity and clinical significance (most significant first)
    """
    
    # Format prior findings if available
    if user_input.priorFindings and len(user_input.priorFindings) > 0:
        prior_findings_info = f"Prior Findings: {', '.join(user_input.priorFindings)}"
    else:
        prior_findings_info = "Prior Findings: None available"
    
    formatted_prompt = imaging_analysis_prompt.format(
        image_type=user_input.imageType.upper(),
        image_url=user_input.imageUrl,
        body_part=user_input.bodyPart,
        clinical_indication=user_input.clinicalIndication,
        patient_age=user_input.patientAge,
        patient_gender=user_input.patientGender,
        prior_findings_info=prior_findings_info,
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
        
        # Validate and create ImagingAnalysis object
        if isinstance(data, dict):
            return ImagingAnalysis(**data)
        else:
            raise ValueError(f"Expected JSON object, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}. Response content: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to process medical imaging analysis response: {str(e)}")

