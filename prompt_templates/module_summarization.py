def module_summarization_prompt(module: dict, supportive_information: str) -> str:

    return f"""

        You are an expert in explainable AI.

        You are given a data for a machine learning model.

        Supportive Information: \n
        
        {supportive_information}
        
        \n
        
        Module: \n
        
        {module}
        
        \n

        - Given the datapoint and model prediction, summarize the results of the module in a concise manner. \n
        - Only use the supportive information as a reference and focus on the results of the module. 
        - Explicitely mention the relevant numbers and values.
        - Use calibrated language (e.g., “likely,” “uncertain,” “tends to”) instead of absolute certainty.
        - Present both strengths and weaknesses (e.g., highlight correct predictions AND typical errors).
        - Clarify limits and avoid exaggerating performance of the module.
        - Clearly state uncertainty and data limitations.
    """
