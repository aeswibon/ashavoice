from pydantic import BaseModel, Field


class SymptomSummary(BaseModel):
    """
    Represents a structured summary of patient symptoms extracted from a transcript.
    """

    chief_complaint: str = Field(
        ..., description="The main reason for the patient's visit."
    )
    history_of_present_illness: str = Field(
        ...,
        description="Detailed description of the chief complaint, including onset, duration, character, severity, location, aggravating/alleviating factors, and associated symptoms.",
    )
    relevant_past_medical_history: str | None = Field(
        None,
        description="Any past medical conditions relevant to the current presentation.",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="List of current medications the patient is taking.",
    )
    allergies: list[str] = Field(
        default_factory=list, description="List of known allergies."
    )
    family_history: str | None = Field(
        None, description="Relevant family medical history."
    )
    patient_needs_expectations: str | None = Field(
        None,
        description="Any explicit needs or expectations stated by the patient regarding the visit.",
    )
