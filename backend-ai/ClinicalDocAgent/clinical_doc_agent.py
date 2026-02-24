"""
Clinical Documentation & Tracking - AI agents per AI Endpoints Specification.
All outputs include: ai_version, generated_at, status (pending_clinician_review).
"""
import sys
import os
import json
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Prompts.prompt import (
    clinical_note_summarize_prompt,
    therapy_session_note_prompt,
    clinical_summary_trends_prompt,
    risk_detection_prompt,
    mental_health_risk_prompt,
    trend_analysis_prompt,
    population_insights_prompt,
    form_fill_prompt,
)
from Configurations.config import llm_model, AI_VERSION
from PydanticModels.model import (
    ClinicalNoteSummarizeInput,
    ClinicalNoteSummarizeOutput,
    TherapySessionNoteInput,
    TherapySessionNoteOutput,
    StructuredTherapyNote,
    ClinicalSummaryTrendsInput,
    ClinicalSummaryTrendsOutput,
    RiskDetectionInput,
    RiskDetectionOutput,
    MentalHealthRiskInput,
    MentalHealthRiskOutput,
    TrendAnalysisInput,
    TrendAnalysisOutput,
    PopulationInsightsInput,
    PopulationInsightsOutput,
    FormFillInput,
    FormFillOutput,
    ICDCodingInput,
    ICDCodingOutput,
    MedicationAdherenceInput,
    AdherencePredictionSpecOutput,
)


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())


def _invoke(prompt: str) -> dict:
    response = llm_model.LLM().invoke(prompt)
    return _parse_json(response.content)


def clinical_note_summarize(inp: ClinicalNoteSummarizeInput) -> ClinicalNoteSummarizeOutput:
    prompt = clinical_note_summarize_prompt.format(
        patient_id=inp.patient_id,
        visit_date=inp.visit_date,
        raw_notes=inp.raw_notes,
        module=inp.module,
    )
    data = _invoke(prompt)
    return ClinicalNoteSummarizeOutput(
        summary_note=data.get("summary_note", ""),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def therapy_session_note(inp: TherapySessionNoteInput) -> TherapySessionNoteOutput:
    prompt = therapy_session_note_prompt.format(
        patient_id=inp.patient_id,
        session_date=inp.session_date,
        session_notes=inp.session_notes,
        module=inp.module,
    )
    data = _invoke(prompt)
    sn = data.get("structured_note", {})
    if isinstance(sn, dict):
        structured = StructuredTherapyNote(
            mood_score=int(sn.get("mood_score", 5)),
            symptoms=sn.get("symptoms", []),
            recommendations=sn.get("recommendations", []),
        )
    else:
        structured = StructuredTherapyNote(mood_score=5, symptoms=[], recommendations=[])
    return TherapySessionNoteOutput(
        structured_note=structured,
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def clinical_summary_trends(inp: ClinicalSummaryTrendsInput) -> ClinicalSummaryTrendsOutput:
    prompt = clinical_summary_trends_prompt.format(
        patient_id=inp.patient_id,
        module=inp.module,
        vitals=json.dumps(inp.vitals),
        labs=json.dumps(inp.labs),
    )
    data = _invoke(prompt)
    return ClinicalSummaryTrendsOutput(
        trend_summary=data.get("trend_summary", ""),
        alerts=data.get("alerts", []),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def risk_detection(inp: RiskDetectionInput) -> RiskDetectionOutput:
    prompt = risk_detection_prompt.format(
        patient_id=inp.patient_id,
        module=inp.module,
        vitals=json.dumps(inp.vitals),
        labs=json.dumps(inp.labs),
        med_adherence=inp.med_adherence,
    )
    data = _invoke(prompt)
    return RiskDetectionOutput(
        risk_level=data.get("risk_level", "moderate"),
        alerts=data.get("alerts", []),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def mental_health_risk(inp: MentalHealthRiskInput) -> MentalHealthRiskOutput:
    prompt = mental_health_risk_prompt.format(
        patient_id=inp.patient_id,
        mood_logs=inp.mood_logs or [],
        therapy_notes=inp.therapy_notes or "",
        symptom_scores=json.dumps(inp.symptom_scores or {}),
    )
    data = _invoke(prompt)
    return MentalHealthRiskOutput(
        risk_level=data.get("risk_level", "moderate"),
        alerts=data.get("alerts", []),
        recommendations=data.get("recommendations", []),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def trend_analysis(inp: TrendAnalysisInput) -> TrendAnalysisOutput:
    prompt = trend_analysis_prompt.format(
        patient_id=inp.patient_id,
        module=inp.module,
        vitals=json.dumps(inp.vitals or {}),
        labs=json.dumps(inp.labs or {}),
        outcomes=inp.outcomes or [],
        reference_ranges=json.dumps(inp.reference_ranges or {}),
    )
    data = _invoke(prompt)
    return TrendAnalysisOutput(
        trend_summary=data.get("trend_summary", ""),
        reference_ranges_used=data.get("reference_ranges_used"),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def population_insights(inp: PopulationInsightsInput) -> PopulationInsightsOutput:
    prompt = population_insights_prompt.format(
        module=inp.module,
        aggregate_metrics=json.dumps(inp.aggregate_metrics),
    )
    data = _invoke(prompt)
    return PopulationInsightsOutput(
        insights=data.get("insights", {}),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def form_fill(inp: FormFillInput) -> FormFillOutput:
    prompt = form_fill_prompt.format(
        patient_id=inp.patient_id,
        form_type=inp.form_type,
        module=inp.module,
        source_data=json.dumps(inp.source_data),
    )
    data = _invoke(prompt)
    return FormFillOutput(
        filled_fields=data.get("filled_fields", {}),
        validation_errors=data.get("validation_errors", []),
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def icd_coding(patient_id: str, module: str, notes: str) -> ICDCodingOutput:
    """Suggest ICD-10/ICD-11 codes based on notes. Wraps ICD10 agent with spec output."""
    from ICD10Agent.icd10_agent import get_icd10_suggestions
    from PydanticModels.model import ICD10Input

    suggestions = get_icd10_suggestions(ICD10Input(diagnosis=notes))
    return ICDCodingOutput(
        suggestions=suggestions,
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )


def adherence_prediction(inp: MedicationAdherenceInput) -> AdherencePredictionSpecOutput:
    """Predict likelihood of follow-up or medication adherence. Wraps Adherence agent with spec output."""
    from AdherenceAgent.adherence_agent import predict_medication_adherence

    pred = predict_medication_adherence(inp)
    return AdherencePredictionSpecOutput(
        adherenceProbability=pred.adherenceProbability,
        riskLevel=pred.riskLevel,
        riskFactors=pred.riskFactors,
        interventions=pred.interventions,
        ai_version=AI_VERSION,
        generated_at=_generated_at(),
        status="pending_clinician_review",
    )
