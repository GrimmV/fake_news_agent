def module_summarization_prompt(modules: str, supportive_information: str) -> str:

    return f"""

        You are an expert in explainable AI.

        You are given a data for a machine learning model.

        Supportive Information: \n
        
        {supportive_information}
        
        \n
        
        Modules: \n
        
        {modules}
        
        \n

        Given the datapoint and model prediction, summarize the results of the modules in a concise manner. \n
        Only use the supportive information as a reference and focus on the results of the modules. Explicitely 
        mention the relevant numbers and values.
        
    """
