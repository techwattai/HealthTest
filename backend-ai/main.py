from Endpoints import body_vitals, ai_appointments, ai_diagnosis, ai_summarization, ai_icd10, ai_drug_interaction, ai_guest_booking, ai_health_analysis, ai_vitals_anomaly, ai_adherence, ai_lab_interpretation, ai_readmission, ai_prescription, ai_no_show, ai_imaging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Hospital Vitals Live ML API",
    description="API to stream live patient vitals and get ML-based health predictions.",
    version="1.0.0"
    
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(body_vitals.router)
app.include_router(ai_appointments.router)
app.include_router(ai_diagnosis.router)
app.include_router(ai_summarization.router)
app.include_router(ai_icd10.router)
app.include_router(ai_drug_interaction.router)
app.include_router(ai_guest_booking.router)
app.include_router(ai_health_analysis.router)
app.include_router(ai_vitals_anomaly.router)
app.include_router(ai_adherence.router)
app.include_router(ai_lab_interpretation.router)
app.include_router(ai_readmission.router)
app.include_router(ai_prescription.router)
app.include_router(ai_no_show.router)
# app.include_router(ai_imaging.router)