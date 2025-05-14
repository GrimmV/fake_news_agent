from typing import List, Dict, Any
def trust_assessment_prompt(trace: List[Dict[str, Any]]) -> str:
    return f"""
    You are an expert in explainable AI.
    You are given a trace of actions conducted on a set of data and module information about the particular prediction of a machine learning model.
    You are to assess the trustworthiness of machine learning model's prediction.
    
    This is the trace of actions:
    {trace}

    How trustworthy do you estimate the prediction to be? Focus on the actions related to local explanations and counterfactuals/similar texts.
    
    Mention the most important numbers and values in your reasoning.
    """

