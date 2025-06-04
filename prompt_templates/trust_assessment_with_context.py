from typing import List, Dict, Any


def trust_assessment_with_context_prompt(
    trace: List[Dict[str, Any]], context: List[str], module_focus: str, statement: str, sceptical: bool = False
) -> str:
    module_prompt = ""
    context_prompt = ""
    if module_focus != "":
        module_prompt = f"The user believes that the module {module_focus} provides the most relevant information about the prediction, therefore, specifically consider this modules information."
    if context != []:
        context_prompt = "The user believes that the following context is relevant for the assessment: "
        for ctx in context:
            context_prompt += f"{ctx}\n"

    return f"""
    You are an expert in explainable AI.
    You are given a trace of actions conducted on a set of data and module information about the particular prediction of a machine learning model.
    You are to assess the trustworthiness of machine learning model's prediction in order to help the user make a judgement call about the trustworthiness of the model.
    
    This is the trace of actions:
    {trace}\\

    {module_prompt}\\
    
    {context_prompt}\\
    
    This is the statement:
    {statement}


    How trustworthy do you estimate the prediction to be? Focus on the actions related to local explanations and counterfactuals/similar texts.
    
    {"Be as sceptical as possible" if sceptical else ""}
    
    Mention the most important numbers and values in your reasoning.
    """
