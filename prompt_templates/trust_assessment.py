from typing import List, Dict, Any
def trust_assessment_prompt(trace: List[Dict[str, Any]], statement: str, sceptical: bool = False) -> str:
    return f"""
    You are an expert in explainable AI.
    You are given a trace of actions conducted on a set of data and module information about the particular prediction of a machine learning model.
    You are to assess the trustworthiness of machine learning model's prediction in order to help the user make a judgement call about the trustworthiness of the model.
    
    This is the trace of actions:
    {trace}
    
    This is the statement:
    {statement}

    How trustworthy do you estimate the prediction to be? Focus on the actions related to local explanations.
    
    {"Be as sceptical as possible" if sceptical else ""}
    
    Mention the most important numbers and values in your reasoning.
    """

