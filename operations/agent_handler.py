import pandas as pd

from pydantic import BaseModel, Field
from typing import List, Dict
from typing_extensions import Annotated
from pydantic import AfterValidator
from enum import Enum

from prompt_templates.initial import initial_prompt
from prompt_templates.assessment import assessment_prompt
from prompt_templates.guidance import guidance_prompt
from prompt_templates.query_classification import query_classification_prompt
from prompt_templates.clarification import clarification_prompt
from prompt_templates.objection import objection_prompt
from prompt_templates.continuation import continuation_prompt
from prompt_templates.continuation2 import continuation_prompt2

from operations.utils.retrieve_datapoint import retrieve_datapoint

class Objection(BaseModel):
    objection: str
    
class Clarification(BaseModel):
    clarification: str

class QueryClass(Enum):
    OUT_OF_SCOPE = "out-of-scope"
    CONTINUATION = "continuation"
    OBJECTION = "objection"
    AMBIGUOUS = "ambiguous"
    OTHER = "other"

class QueryClassification(BaseModel):
    query_class: QueryClass = Field(
        description="Correctly classify the user request"
    )
    explanation: str
    

class NextSteps(BaseModel):
    suggestion1: str
    suggestion2: str
    suggestion3: str

class XaiInsights(BaseModel):
    observations: str
    conclusions: str
    critical_reflection: str
    
class XaiInsights2(BaseModel):
    observations: str
    conclusions: str

class ModuleChoice(BaseModel):
    module: str
    parameters: Dict[str, str]
    explanation: str
    
def max_three_modules(v: List[ModuleChoice]) -> str:
    print("#################################")
    print("chosen modules:")
    print(v)
    if len(v) > 6:
        raise ValueError("The number of modules must not exceed 3")
    return v

class Modules(BaseModel):
    modules: Annotated[List[ModuleChoice], AfterValidator(max_three_modules)]
    
class AgentHandler:
    
    def __init__(self, llm, label_descriptions: dict = None, feature_descriptions: dict = None, module_descriptions: dict = None):
        self.df = pd.read_csv("data/full_df.csv")
        self.llm = llm
        self.label_descriptions = label_descriptions
        self.feature_descriptions = feature_descriptions
        self.module_descriptions = module_descriptions
        
    
    def get_relevant_modules(self, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = initial_prompt(self.module_descriptions, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=Modules,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def compute_insights(self, modules, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = assessment_prompt(modules, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=XaiInsights,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_insights2(self, modules, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = assessment_prompt(modules, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=XaiInsights2,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def compute_next_steps(self, history, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = guidance_prompt(self.module_descriptions, datapoint, history, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=NextSteps,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def classify_query(self, request, history, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = query_classification_prompt(request, history, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=QueryClassification,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def clarify(self, request, history, modules, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = clarification_prompt(request, history, modules, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=Clarification,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def objection(self, request, history, modules, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = objection_prompt(request, history, modules, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=Objection,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def continuation(self, request, history, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = continuation_prompt(request, history, self.module_descriptions, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=Modules,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def continuation2(self, request, history, dp_id) -> dict:
            
        datapoint = retrieve_datapoint(self.df, dp_id)
        
        prompt = continuation_prompt2(request, history, self.module_descriptions, datapoint, self.label_descriptions, self.feature_descriptions)
        
        response = self.llm.generate(
            prompt,
            response_model=ModuleChoice,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()