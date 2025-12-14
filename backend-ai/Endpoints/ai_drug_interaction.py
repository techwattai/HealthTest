import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DrugInteractionAgent.drug_interaction_agent import check_drug_interactions
from PydanticModels.model import DrugInteractionInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-drug-interaction", tags=["AI Drug Interaction"])
def drug_interaction_endpoint(user_input: DrugInteractionInput):
    """
    Endpoint to check for potential drug interactions when multiple medications are prescribed.
    
    Input:
    - drugs: Array of medication names (List[str])
    
    Output:
    - Array of drug interaction objects with:
      - severity: 'low' | 'moderate' | 'high' | 'severe'
      - msg: Interaction description (string)
      - drugs: Affected drugs (List[str])
      - recommendation: Clinical recommendation (optional string)
    """
    try:
        interactions = check_drug_interactions(user_input)
        return interactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing drug interactions: {str(e)}")

