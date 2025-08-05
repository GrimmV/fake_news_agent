TECHNICAL_CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with explainable AI data (such as SHAP values, feature 
importance, or other interpretability metrics) and a natural language description that 
claims to explain this data. Your objective is to evaluate whether the natural language 
description is clear and accurate in representing the underlying explainable AI data.

Your response should be a single word: either "clear" or "unclear," and it should not include any other 
text or characters. "clear" indicates that the natural language description is well-structured, easy to understand, and 
appropriately addresses the explainable AI data. "unclear" indicates that the natural language description is ambiguous, poorly organized, or 
not effectively communicated. Please carefully consider the explainable AI data and the natural language description before determining your 
response.

After analyzing the explainable AI data and the natural language description, you must write a detailed explanation of your reasoning to 
justify why you chose either "clear" or "unclear." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the natural language description does or does not meet the 
criteria for clarity.

[BEGIN DATA]
Used function: {function}
Use params: {params}
Resulting Explainable AI Data: {module_output}
Natural Language Description: {natural_language_description}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the clarity of the natural language description based on the explainable AI data.
LABEL: "clear", "mostly clear" or "unclear"
"""