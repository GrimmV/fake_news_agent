from prompt_templates.base import base_prompt

def initial_assessment2_prompt(module_list, datapoint):
    
    module_data = ""
    
    for elem in module_list:
        module_data = f'''{module_data}{elem["name"]}: \n
        parameters: {elem["params"]}\n
        data: {elem["data"]}\n\n
    '''
    
    base = base_prompt(datapoint)

    prompt = f'''{base} \\
        
    At this time, the following data is shown to the user: \\
    
    {module_data}\\
    
    Carefully analyze the data and describe the most relevant observations and the conclusions you can draw from 
    them. Focus on the provided XAI data in your analysis, rather than the datapoint itself.\\
    '''

    return prompt