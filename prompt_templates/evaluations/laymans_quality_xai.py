LAYMAN_CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a technical description and a layman's description that claims to explain the same concept in simpler terms. Your objective is to evaluate whether the layman's description is truthful and accurate in representing the underlying technical description.

A truthful layman's description is one that accurately translates the technical concepts into simpler language without misrepresentation, exaggeration, or omission of important details. An untruthful layman's description is one that misinterprets, distorts, or misrepresents the technical information, even if it may sound plausible or well-written.

Your response should be a single word: either "truthful" or "untruthful," and it should not include any other text or characters. "truthful" indicates that the layman's description accurately represents the technical description. "untruthful" indicates that the layman's description misrepresents, distorts, or inaccurately describes the technical information.

After analyzing the technical description and the layman's description, you must write a detailed explanation of your reasoning to justify why you chose either "truthful" or "untruthful." Avoid stating the final label at the beginning of your explanation. Your reasoning should include specific points about how the layman's description does or does not accurately represent the technical description.

[BEGIN DATA]
Technical Description: {technical_description}
Layman's Description: {layman_description}
[END DATA]
Please analyze the technical description carefully and compare it with the layman's description to assess truthfulness.

EXPLANATION: Provide your reasoning step by step, evaluating whether the layman's description accurately represents the technical description.
LABEL: "truthful" or "untruthful"
"""