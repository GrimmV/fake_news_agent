def initial_assessment_prompt(module_list, datapoint):
    
    module_data = ""
    
    for elem in module_list:
        module_data = f'''{module_data}{elem["name"]}: \n
        description: {elem["description"]}\n
        parameters: {elem["parameters"]}\n
        data: {elem["data"]}\n\n
    '''

    prompt = f'''A Machine Learning Model has been trained to predict if a given social media post 
    contains fake information or not. \\
    
    The end user wants to dive deeper into the decision process of 
    the model to make a judgement on the correctness of the particular ML model prediction. \\
        
    The user is supposed to be guided through a semi-structured process that makes sure that they 
    have understood the general workings of the model, the structure and origin of the data, the 
    reasons for the particular prediction and the context around the prediction. \\
                
    The following is the datapoint and the models output: \\
    
    Post content: {datapoint["text"]} \\
    Properties: {datapoint["properties"]} \\
    Model prediction: {datapoint["prediction"]} \\
    Model probabilities: {datapoint["probas"]} \\
        
    At this time, the following data is shown to the user: \\
    
    {module_data}\\
    
    Carefully analyze the data and discuss the most relevant insights that help the user 
    to get a firm understanding of the models behaviour and how it arrived at the prediction
    '''

    return prompt