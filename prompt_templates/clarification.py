from base import base_prompt

def clarification_prompt(request, history, module_list, datapoint):
    
    module_data = ""
    
    for elem in module_list:
        module_data = f'''{module_data}{elem["name"]}: \n
        parameters: {elem["params"]}\n
        data: {elem["data"]}\n\n
    '''

    
    base = base_prompt(datapoint)

    prompt = f'''{base} \\
        
    This is a summary of the most recent conversation history:  \\
    
    {history} \\
        
    Currently, the following data is shown to the user: \\
    
    {module_data}\\
        
    The user made a request of clarification towards the currently displayed visualizations and your assessment: \\

    {request}
    '''

    return prompt