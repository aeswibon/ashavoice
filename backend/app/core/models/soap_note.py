from pydantic import BaseModel, Field


class SOAPNote(BaseModel):
    """
    Represents a SOAP (Subjective, Objective, Assessment, Plan) note.
    """

    subjective: str = Field(
        ...,
        description="Patient's subjective description of their symptoms and concerns.",
    )
    objective: str = Field(
        ...,
        description="Objective findings (e.g., from physical exam, lab results - will be inferred from patient statements for now).",
    )
    assessment: str = Field(
        ...,
        description="Healthcare professional's assessment of the patient's condition/diagnosis.",
    )
    plan: str = Field(
        ...,
        description="Proposed course of action, including treatments, referrals, and follow-up.",
    )
