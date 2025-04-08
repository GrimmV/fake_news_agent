import json

def query_classification_prompt(request, history, datapoint, label_descriptions = {}, feature_descriptions = []):

    prompt = f'''A Machine Learning Model has been trained to predict if a given social media post 
    contains fake information or not based on the post content and some properties of it. \\
    
    The end user wants to dive deeper into the decision process of the model to make a judgement on 
    the correctness of the particular ML model prediction and build trust towards it. \\
        
    The following is the datapoint and the models output: \\
    
    Post content: {datapoint["statement"]} \\
    Properties: {datapoint["properties"]} \\
    Model prediction: {datapoint["prediction"]} \\  
            
    These are the possible classes:\\
    
    {json.dumps(label_descriptions)}\\
    
    These are the features that the model uses additionally to the statement: \\
        
    {json.dumps(feature_descriptions)}\\
                
    This is a summary of the most recent conversation history:  \\
    
    {history} \\

    The most recent request of the user is: \\

    {request} \\

    Categorize the users request into one of the following categories: \\
        out-of-scope: the request is not relevant for fake news classification in social media, \\
        clarification: The user asks for clarification of concepts, observations or insights, \\
        confirmation: The user approves the current state of the conversation and/or wants to move on with the analysis, e.g. based on the suggested next steps, \\
        objection: The user is challenging insights from the assistant, \\
        ambiguous: The user request is not clear enough to handle it, \\
        other: None of the above \\
    '''

    return prompt