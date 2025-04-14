from base import base_prompt

def assessment_prompt(module_list, datapoint):
    
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
    
    Carefully analyze the data, describe the most relevant observations, the conclusions you can draw from 
    them and critically reflect on your analysis, highlighting the most relevant limitations. Focus on the 
    provided XAI data in your analysis, rather than the datapoint itself.
    '''

    return prompt