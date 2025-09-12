from dataclasses import dataclass
from typing import Dict, Any, Optional
from divergence_analyzer import DivergenceAnalyzer


@dataclass
class DivergenceThresholds:
    """Configurable thresholds for investigation logic."""
    # 'Substantial' disagreement between label and prediction:
    divergence_threshold: int = 1          # e.g., |label - prediction| >= 2
    # 'Close' agreement between label and prediction:
    alignment_threshold: int = 0           # e.g., |label - prediction| <= 1
    # Trust thresholds on 0..3 scale
    trust_high_threshold: int = 2          # High trust (LLM is confident)
    trust_low_threshold: int = 1           # Low trust (LLM is skeptical)


def conduct_divergence_analysis(trace: str,
                                case_type: str,
                                label: int,
                                prediction: int,
                                trustscore: int) -> Dict[str, Any]:
    
    analyzer = DivergenceAnalyzer(model_name="gpt-5-2025-08-07")
    analysis = analyzer.analyze_trace(trace, case_type, label, prediction, trustscore)
    
    
    return {
        "case_type": case_type,  # "model_wrong_llm_trusts" or "model_right_llm_distrusts"
        "label": label,
        "prediction": prediction,
        "trustscore": trustscore,
        "analysis": analysis
    }


def assess_case(trace: str,
                label: int,
                prediction: int,
                trustscore: int,
                thresholds: Optional[DivergenceThresholds] = None,
               ) -> Dict[str, Any]:
    """
    Decide whether a case should be investigated based on:
      - Ground-truth 'label' (0..5)
      - ML 'prediction' (0..5)
      - LLM 'trustscore' (0..3)

    Investigation rules (defaults shown in DivergenceThresholds):
      - Investigate Type A (model_wrong_llm_trusts) if |label - prediction| >= divergence_threshold and trustscore >= trust_high_threshold
      - Investigate Type B (model_right_llm_distrusts) if |label - prediction| <= alignment_threshold and trustscore <= trust_low_threshold

    Returns:
      dict with keys:
        - should_investigate: bool
        - case_type: str | None
        - rationale: str
        - analysis: dict | None   (output of 'analyzer' when triggered)
    """
    th = thresholds or DivergenceThresholds()
    gap = abs(label - prediction)

    # Type A: Model appears wrong, but LLM is confident in trusting it
    if gap >= th.divergence_threshold and trustscore >= th.trust_high_threshold:
        case_type = "model_wrong_llm_trusts"
        rationale = (f"Substantial label–prediction gap (|{label}-{prediction}|={gap:.2f} ≥ {th.divergence_threshold}) "
                     f"AND high trustscore ({trustscore} ≥ {th.trust_high_threshold}).")
        return {
            "should_investigate": True,
            "case_type": case_type,
            "rationale": rationale,
            "analysis": conduct_divergence_analysis(trace, case_type, label, prediction, trustscore)
        }

    # Type B: Model appears right, but LLM is skeptical
    if gap <= th.alignment_threshold and trustscore <= th.trust_low_threshold:
        case_type = "model_right_llm_distrusts"
        rationale = (f"Close label–prediction alignment (|{label}-{prediction}|={gap:.2f} ≤ {th.alignment_threshold}) "
                     f"AND low trustscore ({trustscore} ≤ {th.trust_low_threshold}).")
        return {
            "should_investigate": True,
            "case_type": case_type,
            "rationale": rationale,
            "analysis": conduct_divergence_analysis(trace, case_type, label, prediction, trustscore)
        }

    # No investigation
    return {
        "should_investigate": False,
        "case_type": None,
        "rationale": (
            f"No trigger: gap={gap:.2f} (thr align≤{th.alignment_threshold}, diverge≥{th.divergence_threshold}), "
            f"trust={trustscore} (low≤{th.trust_low_threshold}, high≥{th.trust_high_threshold})."
        ),
        "analysis": None
    }


# --- Example usage ---
if __name__ == "__main__":
    # Case B example: model and label close, but LLM distrusts
    print(assess_case(trace="", label=0, prediction=0, trustscore=0))

    # Case A example: model and label far apart, but LLM highly trusting
    print(assess_case(trace="", label=0, prediction=4, trustscore=2))

    # Neutral example: no trigger
    print(assess_case(trace="", label=2, prediction=3, trustscore=1))
