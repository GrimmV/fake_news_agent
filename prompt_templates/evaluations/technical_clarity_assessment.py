TECHNICAL_ASSESSMENT_CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with natural language descriptions of explainable AI data (such as SHAP values, feature 
importance, or other interpretability metrics) and a technical assessment that claims to assess these descriptions and make a judgement call. 
Your objective is to evaluate whether the technical assessment is clear and accurate in representing the underlying natural language descriptions.

Your response should be a single word: either "clear" or "unclear," and it should not include any other 
text or characters. "clear" indicates that the technical assessment is well-structured, easy to understand, and 
appropriately addresses the natural language descriptions. "unclear" indicates that the technical assessment is ambiguous, poorly organized, or 
not effectively communicated. Please carefully consider the natural language descriptions and the technical assessment before determining your 
response.

After analyzing the natural language descriptions and the technical assessment, you must write a detailed explanation of your reasoning to 
justify why you chose either "clear" or "unclear." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the technical assessment does or does not meet the 
criteria for clarity.

[BEGIN DATA]
Natural Language Descriptions: {natural_language_descriptions}
Technical Assessment: {assessment}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the clarity of the technical assessment based on the natural language descriptions.
LABEL: "clear" or "unclear"
"""