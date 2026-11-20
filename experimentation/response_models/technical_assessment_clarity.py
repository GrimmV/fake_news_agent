from pydantic import BaseModel, Field
from typing import Literal

class TechnicalAssessmentClarity(BaseModel):
    explanation: str = Field(..., description="Explanation for the chosen level of truthfulness.")
    clarity: Literal["clear", "mostly clear", "unclear"] = Field(..., description="The level of clarity.")