from pydantic import BaseModel, Field
from typing import Literal

class FocusQuality(BaseModel):
    explanation: str = Field(..., description="Explanation for the chosen level of focus quality.")
    focus_quality: Literal["strongly contributes", "weakly contributes"] = Field(..., description="The level of focus quality.")