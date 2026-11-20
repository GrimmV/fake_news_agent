from pydantic import BaseModel, Field
from typing import Literal

class ShortAssessmentTruthfulness(BaseModel):
    explanation: str = Field(..., description="Explanation for the chosen level of truthfulness.")
    truthfulness: Literal["truthful", "mostly truthful", "untruthful"] = Field(..., description="The level of truthfulness.")