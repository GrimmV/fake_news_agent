FOCUS_QUALITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with natural language descriptions of explainable AI data (such as SHAP values, feature 
importance, or other interpretability metrics) and a technical assessment that claims to assess these descriptions and make a judgement call. 
Your objective is to evaluate whether the underlying natural language descriptions significantly contribute to the technical assessment.

Your response should be a single word: either "strongly contributes" or "weakly contributes" and it should not include any other 
text or characters. "strongly contributes" indicates that the natural language descriptions significantly contribute to the technical assessment. "weakly contributes" indicates that the natural language descriptions do not significantly contribute to the technical assessment.

After analyzing the natural language descriptions and the technical assessment, you must write a detailed explanation of your reasoning to 
justify why you chose either "strongly contributes" or "weakly contributes." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the natural language descriptions do or do not contribute to the technical assessment.

[BEGIN DATA]
Natural Language Descriptions: {natural_language_descriptions}
Technical Assessment: {assessment}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the contribution of the natural language descriptions to the technical assessment.
LABEL: "strongly contributes" or "weakly contributes"
"""