import instructor
from pydantic import BaseModel
from openai import OpenAI
from dataclasses import dataclass
from llm.logger import log_kwargs, log_exception


@dataclass
class LLMOptions:
    temperature: float = 0.0
    n: int = 1
    top_p: float = 1.0
    max_tokens: int = 150


class GPTModel:
    def __init__(self, model_name, key):
        self.model_name = model_name
        self.key = key
        self.client = instructor.from_openai(OpenAI(api_key=key))
        self.client.on("completion:kwargs", log_kwargs)
        self.client.on("completion:error", log_exception)

    def generate(
        self,
        prompt: str,
        response_model: BaseModel,
        max_retries: int = 3,
        validation_context: dict = None,
        system_message: str = "You are a helpful assistant.",
    ):

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": prompt},
        ]
        gpt_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_retries=max_retries,
            response_model=response_model,
            validation_context=validation_context,
        )

        return gpt_response
