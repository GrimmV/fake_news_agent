from prompt_templates.base import base_prompt


def query_classification_prompt(
    request, available_modules, all_modules, history, datapoint
):

    base = base_prompt(datapoint)

    module_data = ""
    
    print("#####################################################")
    print(available_modules)

    for elem in available_modules:
        module_data = f"""{module_data}{elem["module"]}: \n
        parameters: {elem["parameters"]}\n\n
    """

    module_overview = ""

    for elem in all_modules:
        module_overview = f"""{module_overview}{elem["name"]}: \n
        description: {elem["description"]}\n
        parameters: {elem["parameters"]}\n\n
    """

    prompt = f"""{base} \\
        
    This is a summary of the most recent conversation history:  \\
    
    {history} \\

    The most recent request of the user is: \\

    {request} \\
        
    These are the modules displayed and available:
    
    {module_data}
    
    These are all the modules that can be accessed by you:
    
    {module_overview} \\

    Make the choice if you need to fetch new data or if the available modules suffice to handle the users request. \\
    """

    return prompt


# clarification: The user asks for clarification of concepts, observations or insights, \\
