from base import base_prompt


def query_classification_prompt(request, history, datapoint):

    base = base_prompt(datapoint)

    prompt = f"""{base} \\
        
    This is a summary of the most recent conversation history:  \\
    
    {history} \\

    The most recent request of the user is: \\

    {request} \\

    Categorize the users request into one of the following categories: \\
        out-of-scope: the request is not relevant for fake news classification in social media, \\
        continuation: The user wants to move on with the analysis, e.g. based on the suggested next steps, \\
        objection: The user is challenging insights from the assistant or asking for clarification, \\
        ambiguous: The user request is not clear enough to handle it, \\
        other: None of the above \\
    """

    return prompt


# clarification: The user asks for clarification of concepts, observations or insights, \\
