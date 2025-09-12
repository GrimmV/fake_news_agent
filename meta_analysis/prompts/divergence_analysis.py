def divergence_analysis_prompt(trace: str, case_type: str, label: int, prediction: int, trustscore: int) -> str:

    model_wrong_llm_trusts = """
        The ML model prediction diverges substantially from the ground truth,
        but the LLM gave a HIGH trustscore (trusted a wrong model prediction).
    """

    model_right_llm_distrusts = """
        The ML model prediction is close to the ground truth,
        but the LLM gave a LOW trustscore (distrusted a correct model prediction).
    """

    type_description = f"""
    {model_wrong_llm_trusts if case_type == "model_wrong_llm_trusts" else model_right_llm_distrusts}
    """

    return f"""
        System:
        You are an expert analyst of LLM behavior in hybrid ML + XAI pipelines.
        Your ONLY task is to analyze WHY the LLM’s trust judgement diverged
        from the ML model’s performance.

        Definitions:
        - label: Ground truth value between 0 (pants on fire) and 5 (true).
        - prediction: ML model output in the same value space.
        - trustscore: LLM-assistant trustworthiness score between 0 (poor) and 3 (excellent).

        Instructions:
        - Assume the ML model’s prediction is always correct. Do not re-evaluate the claim itself.
        - Identify ONE primary cause from DivergenceType and optional secondary causes.
        - Base your analysis ONLY on evidence contained in the provided trace and conclusion.
        - Do not invent information that is not present in the trace.
        - Always output valid JSON strictly following the DivergenceAnalysis schema.

        User:
        Here is the case to analyze:
        Case description: {type_description}
        Label: {label}
        Prediction: {prediction}
        Trustscore: {trustscore}

        <TRACE>
        {trace}
        </TRACE>
    """
