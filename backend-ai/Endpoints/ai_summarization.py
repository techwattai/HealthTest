import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SummarizationAgent.summarization_agent import summarize_notes
from PydanticModels.model import NotesSummarizationInput
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.post("/ai-summarization", tags=["AI Summarization"])
def summarization_endpoint(user_input: NotesSummarizationInput):
    """
    Endpoint to transform raw clinical notes into structured, formatted medical documentation.
    
    Input:
    - notes: Raw clinical notes/observations (string)
    
    Output:
    - summary: Structured summary in medical format (string)
    - confidence: Confidence score (0-1) (float)
    """
    try:
        summarized = summarize_notes(user_input)
        return summarized
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing summarization: {str(e)}")

