TRUTHWORTHINESS_LLM_JUDGE_PROMPT = """
In this task, you will be presented with explainable AI data (such as SHAP values, feature importance, or other interpretability metrics) and a natural language description that claims to explain this data. Your objective is to evaluate whether the natural language description is truthful and accurate in representing the underlying explainable AI data.

A truthful description is one that accurately reflects the patterns, relationships, and insights present in the explainable AI data without misrepresentation, exaggeration, or omission of important details. An untruthful description is one that misinterprets, distorts, or misrepresents the data, even if it may sound plausible or well-written.

Your response should be a single word: either "truthful" or "untruthful," and it should not include any other text or characters. "truthful" indicates that the natural language description accurately represents the explainable AI data. "untruthful" indicates that the description misrepresents, distorts, or inaccurately describes the data.

After analyzing the explainable AI data and the natural language description, you must write a detailed explanation of your reasoning to justify why you chose either "truthful" or "untruthful." Avoid stating the final label at the beginning of your explanation. Your reasoning should include specific points about how the description does or does not accurately represent the data.

[BEGIN DATA]
Used function: {function}
Use params: {params}
Resulting Explainable AI Data: {module_output}
Natural Language Description: {natural_language_description}
[END DATA]
Please analyze the explainable AI data carefully and compare it with the natural language description to assess truthfulness.

EXPLANATION: Provide your reasoning step by step, evaluating whether the natural language description accurately represents the explainable AI data.
LABEL: "truthful" or "untruthful"
"""