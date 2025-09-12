from enum import Enum
from typing import Annotated, List
from pydantic import BaseModel, Field, StringConstraints

type_descriptions = """
    Pick ONE primary cause for why the LLM's trust judgement diverged
    from a correct ML prediction:

    - performance_baseline: Distrust due to weak global metrics (acc/F1/confusion).
    - feature_interpretation_bias: Mis/over-weighting risky local features as evidence against the model.
    - explanation_framing: Alarmist/critical wording in explanations primes distrust.
    - label_trust_mismatch: Confuses label meaning with trust (e.g., 'extreme claim' => distrust model).
    - overgeneralization_from_dataset_statistics: Misapplies aggregate dists/histograms to a single case.
    - meta_performance_overweighting: Demands unrealistically high certainty despite correctness.
    - content_model_confusion: Judges the claim itself, not model trustworthiness.
    - other: Residual (prompt misread, hallucination, pipeline error).
"""


class DivergenceType(str, Enum):
    performance_baseline = "performance_baseline"
    feature_interpretation_bias = "feature_interpretation_bias"
    explanation_framing = "explanation_framing"
    label_trust_mismatch = "label_trust_mismatch"
    overgeneralization_from_dataset_statistics = (
        "overgeneralization_from_dataset_statistics"
    )
    meta_performance_overweighting = "meta_performance_overweighting"
    content_model_confusion = "content_model_confusion"
    other = "other"

class Evidence(BaseModel):
    explanation: Annotated[str, StringConstraints(max_length=120)] = Field(
        ...,
        description="short, evidence-backed bullet points (≤120 chars each) explaining WHAT in the trace led to divergence.",
    )
    reference: str = Field(
        ...,
        description="compact pointer into the trace/conclusion",
        example="individual feature importance:summary",
    )
    associated_divergence_type: DivergenceType = Field(
        ...,
        description="The divergence type that this evidence is associated with",
    )


class DivergenceAnalysis(BaseModel):
    """
    Minimal, aggregatable analysis of LLM–ML divergences.
    """

    primary_cause: DivergenceType = Field(
        ...,
        description=f"Primary cause for the divergence: {type_descriptions}",
    )

    secondary_causes: Annotated[
        List[DivergenceType], Field(min_length=0, max_length=4)
    ] = Field(
        default=[],
        description="Secondary causes for the divergence",
    )

    factors: Annotated[List[Evidence], Field(min_length=1, max_length=5)] = Field(
        ..., description=("1–5 points of evidence for the divergence")
    )

    meta_comment: str = Field(
        ...,
        description="≤4 sentences stitching the factors to the evidence in plain language (no new claims).",
    )
