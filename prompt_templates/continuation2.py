from prompt_templates.base import base_prompt

def continuation_prompt2(
    request,
    history,
    module_list,
    datapoint
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
        
    The user made the following request: \\

    {request} \\
        
    These are the available modules that can be used to assist the user in their 
    assessment of the model prediction:  \\
    
    {module_overview} \\
        
    Choose one module provided with its respective parameters and add an 
    explanation for your choice.
    '''
    # Add an explanation for your choice.

    return prompt
