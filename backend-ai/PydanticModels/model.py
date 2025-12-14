from pydantic import BaseModel
from typing import List, Literal, Optional, Union
from datetime import datetime
from uuid import UUID


class UserSymptoms(BaseModel):
    symptoms: List[str]
    user_description: str

class PossibleCauses(BaseModel):
    urgency_level: str
    possible_conditions: List[str]
    recommended_department: str
    summary: str
    confidence_score: float

class DiagnosisInput(BaseModel):
    symptoms: List[str]  # Array of symptom descriptions

class DiagnosisOutput(BaseModel):
    diagnosis: str      # Primary diagnosis name
    icd10: str         # ICD-10 code
    confidence: int    # Percentage (0-100)

class DiagnosisResponse(BaseModel):
    diagnoses: List[DiagnosisOutput]

class NotesSummarizationInput(BaseModel):
    notes: str  # Raw clinical notes/observations

class SummarizedNotes(BaseModel):
    summary: str      # Structured summary in medical format
    confidence: float  # Confidence score (0-1)

class ICD10Input(BaseModel):
    diagnosis: str  # Diagnosis description

class ICD10Suggestion(BaseModel):
    code: str          # ICD-10 code
    desc: str          # Code description
    confidence: int    # Confidence percentage (0-100)

class DrugInteractionInput(BaseModel):
    drugs: List[str]  # Array of medication names

class DrugInteraction(BaseModel):
    severity: Literal['low', 'moderate', 'high', 'severe']
    msg: str  # Interaction description
    drugs: List[str]  # Affected drugs
    recommendation: Optional[str] = None  # Clinical recommendation

class GuestBookingPredictionInput(BaseModel):
    symptoms: List[str]           # Tagged symptoms
    user_description: str         # Free-text symptom description
    age: Optional[int] = None     # Patient age
    gender: Optional[str] = None  # Patient gender
    medical_history: Optional[List[str]] = None  # Known conditions

class AIPrediction(BaseModel):
    urgency_level: Literal["Normal", "High", "Emergency"]
    possible_conditions: List[str]
    recommended_department: str
    summary: str
    confidence_score: float  # 0-1

class VitalsInput(BaseModel):
    bloodPressure: str   # "120/80"
    heartRate: str       # "75"
    temperature: str     # "98.6"
    oxygenSat: str      # "98"

class HealthAnalysisInput(BaseModel):
    age: str
    gender: str
    symptoms: str  # Comma-separated symptoms
    vitals: VitalsInput
    medicalHistory: Optional[List[str]] = None

class Condition(BaseModel):
    name: str
    probability: int        # 0-100
    severity: Literal["mild", "moderate", "severe"]
    description: str

class RecommendedDoctor(BaseModel):
    name: str
    specialty: str
    match: int              # 0-100
    availability: str
    experience: str
    rating: float           # 0-5

class ComprehensiveHealthAnalysis(BaseModel):
    conditions: List[Condition]
    recommendedDoctors: List[RecommendedDoctor]
    remedies: List[str]
    urgency: Literal["routine", "urgent", "emergency"]
    confidence: int           # 0-100
    riskFactors: List[str]
    followUpRecommendations: List[str]

# Vital Signs Anomaly Detection Models
class BloodPressureInput(BaseModel):
    systolic: float
    diastolic: float

class VitalsAnomalyInputData(BaseModel):
    heartRate: Optional[float] = None
    bloodPressure: Optional[BloodPressureInput] = None
    temperature: Optional[float] = None  # Celsius
    oxygenSaturation: Optional[float] = None  # Percentage
    respiratoryRate: Optional[float] = None

class VitalRange(BaseModel):
    min: float
    max: float

class BloodPressureBaseline(BaseModel):
    systolic: float
    diastolic: float

class PatientBaseline(BaseModel):
    heartRate: Optional[VitalRange] = None
    bloodPressure: Optional[BloodPressureBaseline] = None

class PatientContext(BaseModel):
    age: int
    conditions: List[str]
    medications: List[str]
    baseline: Optional[PatientBaseline] = None

class VitalsAnomalyInput(BaseModel):
    patientId: str  # UUID as string for flexibility
    timestamp: str  # ISODateString
    vitals: VitalsAnomalyInputData
    patientContext: PatientContext

class ExpectedRange(BaseModel):
    min: float
    max: float

class VitalAnomaly(BaseModel):
    vitalSign: str
    currentValue: float  # Numeric value - for BP, use separate anomalies for systolic/diastolic
    expectedRange: ExpectedRange
    deviationScore: float  # How far from normal (0-1)
    trendDirection: Literal["stable", "improving", "worsening"]

class VitalsAnomalyDetection(BaseModel):
    isAnomaly: bool
    severity: Literal["low", "medium", "high", "critical"]
    anomalies: List[VitalAnomaly]
    recommendations: List[str]
    alertLevel: Literal["none", "monitor", "notify_nurse", "notify_doctor", "emergency"]
    confidence: float  # 0-1

# Medication Adherence Prediction Models
class Demographics(BaseModel):
    age: int
    socioeconomicStatus: Optional[Literal["low", "medium", "high"]] = None
    education: Optional[str] = None
    employmentStatus: Optional[str] = None

class Prescription(BaseModel):
    medicationCount: int
    dosesPerDay: int
    complexity: int  # 1-10 scale
    duration: int  # Days
    cost: Optional[float] = None

class MedicationHistory(BaseModel):
    previousAdherenceRate: Optional[float] = None  # 0-100
    missedAppointments: Optional[int] = None
    hasSupport: Optional[bool] = None

class MedicationAdherenceInput(BaseModel):
    patientId: str  # UUID as string for flexibility
    demographics: Demographics
    prescription: Prescription
    history: MedicationHistory

class RiskFactor(BaseModel):
    factor: str
    impact: int  # 0-100
    description: str

class Intervention(BaseModel):
    strategy: str
    expectedImprovement: int  # 0-100
    priority: Literal["low", "medium", "high"]

class AdherencePrediction(BaseModel):
    adherenceProbability: int  # 0-100
    riskLevel: Literal["low", "moderate", "high", "very_high"]
    riskFactors: List[RiskFactor]
    interventions: List[Intervention]

# Lab Result Interpretation Models
class ReferenceRange(BaseModel):
    min: float
    max: float

class LabResult(BaseModel):
    testName: str
    value: float
    unit: str
    referenceRange: ReferenceRange

class ClinicalContext(BaseModel):
    symptoms: List[str]
    currentDiagnoses: List[str]
    medications: List[str]
    age: int
    gender: str

class LabInterpretationInput(BaseModel):
    patientId: str  # UUID as string for flexibility
    labResults: List[LabResult]
    clinicalContext: ClinicalContext

class AbnormalFinding(BaseModel):
    test: str
    significance: Literal["critical", "high", "moderate", "low"]
    clinicalImplications: List[str]
    possibleCauses: List[str]
    recommendedActions: List[str]

class SuggestedFollowUp(BaseModel):
    test: str
    reason: str
    urgency: Literal["immediate", "within_24h", "within_week", "routine"]

class LabInterpretation(BaseModel):
    summary: str
    abnormalFindings: List[AbnormalFinding]
    suggestedFollowUp: List[SuggestedFollowUp]
    confidence: float  # 0-1

# Readmission Risk Prediction Models
class ReadmissionDemographics(BaseModel):
    age: int
    gender: str
    insurance: str
    socialSupport: Literal["none", "limited", "moderate", "strong"]

class ClinicalData(BaseModel):
    primaryDiagnosis: str
    comorbidities: List[str]
    lengthOfStay: int
    previousAdmissions: int
    emergencyVisits: int

class DischargeInfo(BaseModel):
    medications: int
    followUpScheduled: bool
    homeHealthOrdered: bool
    patientEducationProvided: bool

class ReadmissionRiskInput(BaseModel):
    patientId: str  # UUID as string for flexibility
    demographics: ReadmissionDemographics
    clinicalData: ClinicalData
    discharge: DischargeInfo

class TopRiskFactor(BaseModel):
    factor: str
    contribution: int  # Percentage contribution to risk (0-100)
    modifiable: bool

class PreventativeIntervention(BaseModel):
    intervention: str
    expectedRiskReduction: int  # Percentage (0-100)
    cost: Literal["low", "medium", "high"]
    priority: int  # 1-10 scale, higher = more priority

class ReadmissionRisk(BaseModel):
    riskScore: int  # 0-100
    riskCategory: Literal["low", "moderate", "high", "very_high"]
    predictedDays: Optional[int] = None  # When readmission likely to occur
    topRiskFactors: List[TopRiskFactor]
    preventativeInterventions: List[PreventativeIntervention]
    confidence: float  # 0-1

# Clinical Decision Support for Prescriptions Models
class KidneyFunction(BaseModel):
    creatinine: float
    gfr: float

class PatientFactors(BaseModel):
    age: int
    weight: Optional[float] = None
    kidneyFunction: Optional[KidneyFunction] = None
    liverFunction: Optional[str] = None
    allergies: List[str]
    currentMedications: List[str]
    comorbidities: List[str]
    pregnancy: Optional[bool] = None

class PrescriptionPreferences(BaseModel):
    costSensitive: Optional[bool] = None
    preferGeneric: Optional[bool] = None
    routePreference: Optional[Literal["oral", "iv", "im", "any"]] = None

class PrescriptionSupportInput(BaseModel):
    diagnosis: str
    patientFactors: PatientFactors
    preferences: Optional[PrescriptionPreferences] = None

class PrimaryRecommendation(BaseModel):
    medication: str
    dose: str
    frequency: str
    duration: str
    route: str
    rationale: str
    evidenceLevel: Literal["A", "B", "C"]
    cost: Literal["low", "medium", "high"]
    sideEffects: List[str]
    monitoring: List[str]

class AlternativeMedication(BaseModel):
    medication: str
    whenToConsider: str
    advantages: List[str]
    disadvantages: List[str]

class DrugInteractionWarning(BaseModel):
    interaction: str
    severity: Literal["low", "moderate", "high"]
    management: str

class PrescriptionRecommendation(BaseModel):
    primaryRecommendations: List[PrimaryRecommendation]
    alternatives: List[AlternativeMedication]
    contraindications: List[str]
    warnings: List[str]
    drugInteractions: List[DrugInteractionWarning]

# Appointment No-Show Prediction Models
class AppointmentDetails(BaseModel):
    type: str
    department: str
    leadTime: int  # Days until appointment
    dayOfWeek: str
    timeOfDay: Literal["morning", "afternoon", "evening"]

class PatientHistory(BaseModel):
    totalAppointments: int
    missedAppointments: int
    lastMinuteCancellations: int
    averageLeadTime: float

class NoShowDemographics(BaseModel):
    age: int
    distance: float  # Miles from facility
    transportationAccess: Literal["own", "public", "limited"]
    employmentStatus: str

class Engagement(BaseModel):
    remindersSent: int
    responsesToReminders: int
    portalActive: bool

class NoShowPredictionInput(BaseModel):
    patientId: str  # UUID as string for flexibility
    appointmentDetails: AppointmentDetails
    patientHistory: PatientHistory
    demographics: NoShowDemographics
    engagement: Engagement

class ContributingFactor(BaseModel):
    factor: str
    weight: float  # 0-1, contribution to no-show risk

class NoShowRecommendation(BaseModel):
    action: str
    expectedImpact: int  # 0-100, percentage reduction in no-show probability
    effort: Literal["low", "medium", "high"]

class NoShowPrediction(BaseModel):
    probability: int  # 0-100
    riskLevel: Literal["low", "moderate", "high"]
    contributingFactors: List[ContributingFactor]
    recommendations: List[NoShowRecommendation]

# Medical Imaging Analysis Models
class Coordinates(BaseModel):
    x: float
    y: float
    width: float
    height: float

class ImagingFinding(BaseModel):
    location: str
    description: str
    severity: Literal["normal", "mild", "moderate", "severe"]
    confidence: float  # 0-1
    coordinates: Optional[Coordinates] = None

class ImagingAnalysisInput(BaseModel):
    imageType: Literal["xray", "ct", "mri", "ultrasound"]
    imageUrl: str
    bodyPart: str
    clinicalIndication: str
    patientAge: int
    patientGender: str
    priorFindings: Optional[List[str]] = None

class ImagingAnalysis(BaseModel):
    findings: List[ImagingFinding]
    impression: str
    recommendations: List[str]
    comparison: Optional[str] = None  # If prior images available
    criticalFindings: bool
    radiologistReviewRequired: bool