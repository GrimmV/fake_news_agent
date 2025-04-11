import json

def assessment_prompt(module_list, datapoint, label_descriptions = {}, feature_descriptions = []):
    
    module_data = ""
    
    for elem in module_list:
        module_data = f'''{module_data}{elem["name"]}: \n
        parameters: {elem["params"]}\n
        data: {elem["data"]}\n\n
    '''

    prompt = f'''A Machine Learning Model has been trained to predict if a given social media post 
    contains fake information or not based on the post content and some properties of it. \\
    
    The end user wants to dive deeper into the decision process of the model to make a judgement on 
    the correctness of the particular ML model prediction and build trust towards it. \\
        
    The following is the datapoint and the models output: \\
    
    Post content: {datapoint["statement"]} \\
    Properties: {[{key: val["value"]} for key, val in datapoint["properties"].items()]} \\
    Model prediction: {datapoint["prediction"]} \\      
            
    These are the possible classes:\\
    
    {json.dumps(label_descriptions)}\\
    
    These are the features that the model uses additionally to the statement: \\
        
    {json.dumps(feature_descriptions)}\\
        
    At this time, the following data is shown to the user: \\
    
    {module_data}\\
    
    Carefully analyze the data and describe the most relevant observations and the conclusions you can draw from 
    them. Focus on the provided XAI data in your analysis, rather than the datapoint itself.
    '''

    return prompt