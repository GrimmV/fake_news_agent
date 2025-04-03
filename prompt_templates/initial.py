import json

def initial_prompt(module_list, datapoint, label_descriptions = {}, feature_descriptions = []):
    
    module_overview = ""
    
    for elem in module_list:
        module_overview = f'''{module_overview}{elem["name"]}: \n
        description: {elem["description"]}\n
        parameters: {elem["parameters"]}\n\n
    '''

    prompt = f'''A Machine Learning Model has been trained to predict if a given social media post 
    contains fake information or not. \\
    
    The end user wants to dive deeper into the decision process of 
    the model to make a judgement on the correctness of the particular ML model prediction. \\
        
    The user is supposed to be guided through a semi-structured process that makes sure that they 
    have understood the general workings of the model, the structure and origin of the data, the 
    reasons for the particular prediction and the context around the prediction. \\
        
    The following is the datapoint and the models output: \\
    
    Post content: {datapoint["statement"]} \\
    Properties: {datapoint["properties"]} \\
    Model prediction: {datapoint["prediction"]} \\
    Model probabilities: {datapoint["probas"]} \\
    
    These are the possible classes:\\
    
    {json.dumps(label_descriptions)}\\
    
    These are the features that the model uses additionally to the statement: \\
        
    {json.dumps(feature_descriptions)}\\
        
    These are the available modules that can be used to assist the user in their 
    assessment of the model prediction:  \\
    
    {module_overview} \\
    
    Choose a maximum of three of the modules provided with their respective parameters 
    and add an explanation for your choice.
    '''

    return prompt