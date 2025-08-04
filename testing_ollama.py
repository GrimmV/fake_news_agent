from llm.llm import OllamaOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("OLLAMA_ENDPOINT")

llm = OllamaOpenAI(model_name=MODEL_NAME, base_url=BASE_URL)

output = llm.generate(
    prompt="What is the capital of France?",
    response_model=str,
)

print(output)







