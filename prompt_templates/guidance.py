from prompt_templates.base import base_prompt

def guidance_prompt(
    module_list, datapoint, history
):

    module_overview = ""

    for elem in module_list:
        module_overview = f"""{module_overview}{elem["name"]}: \n
        description: {elem["description"]}\n
        parameters: {elem["parameters"]}\n\n
    """

    base = base_prompt(datapoint)

    prompt = f'''{base} \\
        
    This is a summary of the most recent conversation history:  \\
    
    {history} \\
        
    These are the available modules that can be used to assist the user in their 
    assessment of the model prediction:  \\
    
    {module_overview} \\
    
    Give the user suggestions on what to explore next based on the history and the possibilities 
    given by the available modules. Provide at least one suggestion that encourages the user to ask a new 
    question to the assistant.
    '''

    return prompt
