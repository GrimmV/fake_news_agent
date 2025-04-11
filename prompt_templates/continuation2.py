import json


def continuation_prompt2(
    request,
    history,
    module_list,
    datapoint,
    label_descriptions={},
    feature_descriptions=[],
):

    module_overview = ""

    for elem in module_list:
        module_overview = f"""{module_overview}{elem["name"]}: \n
        description: {elem["description"]}\n
        parameters: {elem["parameters"]}\n\n
    """

    prompt = f"""A Machine Learning Model has been trained to predict if a given social media post 
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
    
    This is a summary of the most recent conversation history:  \\
    
    {history} \\
        
    The user made the following request: \\

    {request} \\
        
    These are the available modules that can be used to assist the user in their 
    assessment of the model prediction:  \\
    
    {module_overview} \\
        
    Choose one module provided with its respective parameters and add an 
    explanation for your choice.
    """
    # Add an explanation for your choice.

    return prompt
