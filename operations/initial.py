from pydantic import BaseModel
from typing import List
from typing_extensions import Annotated
from pydantic import AfterValidator
from prompt_templates.initial import initial_prompt


class ModuleChoice(BaseModel):
    module: str
    explanation: str
    parameters: List[str]
    
def max_three_modules(v: List[ModuleChoice]) -> str:
    if len(v) > 3:
        raise ValueError("The number of modules must not exceed 3")
    return v

class Modules(BaseModel):
    modules: Annotated[List[ModuleChoice], AfterValidator(max_three_modules)]
    
def get_relevant_modules(module_list, datapoint, llm, label_descriptions = {}, feature_descriptions = []) -> dict:
        
    prompt = initial_prompt(module_list, datapoint, label_descriptions, feature_descriptions)
    
    response = llm.generate(
        prompt,
        response_model=Modules,
        system_message="Your are an expert in explainable AI.",
    )

    return response.dict()