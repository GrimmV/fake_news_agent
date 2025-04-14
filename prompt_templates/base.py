import json
from prompt_templates.utils.features import features
from prompt_templates.utils.labels import labels

def base_prompt(datapoint): 
    
    prompt =  f"""A Machine Learning Model has been trained to predict if a given social media post 
    contains fake information or not based on the post content and some properties of it. \\
    
    The end user wants to dive deeper into the decision process of the model to make a judgement on 
    the correctness of the particular ML model prediction and build trust towards it. \\
        
    The following is the datapoint and the models output: \\
    
    Post content: {datapoint["statement"]} \\
    Properties: {[{key: val["value"]} for key, val in datapoint["properties"].items()]} \\
    Model prediction: {datapoint["prediction"]} \\      
            
    These are the possible classes:\\
    
    {json.dumps(labels)}\\
    
    These are the features that the model uses for the classification additionally to the statement: \\
        
    {json.dumps(features)}\\"""
    
    return prompt
