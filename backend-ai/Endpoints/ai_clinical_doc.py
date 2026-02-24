"""
Clinical Documentation & Tracking - AI endpoints per AI Endpoints Specification.
All responses include: ai_version, generated_at, status (pending_clinician_review).
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import APIRouter, HTTPException
from PydanticModels.model import (
    ClinicalNoteSummarizeInput,
    TherapySessionNoteInput,
    ClinicalSummaryTrendsInput,
    RiskDetectionInput,
    MentalHealthRiskInput,
    TrendAnalysisInput,
    PopulationInsightsInput,
    FormFillInput,
    ICDCodingInput,
    MedicationAdherenceInput,
)
from ClinicalDocAgent.clinical_doc_agent import (
    clinical_note_summarize,
    therapy_session_note,
    clinical_summary_trends,
    risk_detection,
    mental_health_risk,
    trend_analysis,
    population_insights,
    form_fill,
    icd_coding,
    adherence_prediction,
)

router = APIRouter(prefix="/ai", tags=["Clinical Documentation & Tracking"])


@router.post("/clinical-note-summarize")
def clinical_note_summarize_endpoint(inp: ClinicalNoteSummarizeInput):
    """Summarize raw clinical notes into structured documentation. Output: summary_note, ai_version, generated_at, status."""
    try:
        return clinical_note_summarize(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/therapy-session-note")
def therapy_session_note_endpoint(inp: TherapySessionNoteInput):
    """Structure mental health therapy session notes and extract symptom scores (mood_score, symptoms, recommendations)."""
    try:
        return therapy_session_note(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clinical-summary-trends")
def clinical_summary_trends_endpoint(inp: ClinicalSummaryTrendsInput):
    """Generate longitudinal summaries for chronic, pediatric, or geriatric patients. Output: trend_summary, alerts."""
    try:
        return clinical_summary_trends(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-detection")
def risk_detection_endpoint(inp: RiskDetectionInput):
    """Detect high-risk patients (chronic conditions, adherence issues). Output: risk_level, alerts."""
    try:
        return risk_detection(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mental-health-risk")
def mental_health_risk_endpoint(inp: MentalHealthRiskInput):
    """Identify risk of relapse or crisis in mental health patients. Integrates mood logs, therapy notes, symptom scores."""
    try:
        return mental_health_risk(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trend-analysis")
def trend_analysis_endpoint(inp: TrendAnalysisInput):
    """Summarize vitals, labs, or outcomes over time. Includes reference ranges and units."""
    try:
        return trend_analysis(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/population-insights")
def population_insights_endpoint(inp: PopulationInsightsInput):
    """Aggregate anonymized patient data for research/public health. PHI must be anonymized before use."""
    try:
        return population_insights(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/form-fill")
def form_fill_endpoint(inp: FormFillInput):
    """Auto-populate hospital forms, insurance claims, or research consent forms. Validates against form schemas."""
    try:
        return form_fill(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/icd-coding")
def icd_coding_endpoint(inp: ICDCodingInput):
    """Suggest ICD-10/ICD-11 codes based on clinical notes. Output includes confidence score for clinician review."""
    try:
        return icd_coding(inp.patient_id, inp.module, inp.notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adherence-prediction")
def adherence_prediction_endpoint(inp: MedicationAdherenceInput):
    """Predict likelihood of follow-up or medication adherence. Predictions must trigger only clinician-approved reminders."""
    try:
        return adherence_prediction(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
