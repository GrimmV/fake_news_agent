import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing_extensions import Annotated
from pydantic import AfterValidator
from enum import Enum

from prompt_templates.initial import initial_prompt
from prompt_templates.initial_assessment import initial_assessment_prompt
from prompt_templates.initial_assessment2 import initial_assessment2_prompt
from prompt_templates.assessment import assessment_prompt
from prompt_templates.assessment2 import assessment_prompt2
from prompt_templates.guidance import guidance_prompt
from prompt_templates.query_classification import query_classification_prompt
from prompt_templates.clarification import clarification_prompt
from prompt_templates.objection import objection_prompt
from prompt_templates.continuation import continuation_prompt
from prompt_templates.module_summarization import module_summarization_prompt
from prompt_templates.trust_assessment import trust_assessment_prompt
from prompt_templates.trust_assessment_with_context import trust_assessment_with_context_prompt
from operations.utils.retrieve_datapoint import retrieve_datapoint


class Objection(BaseModel):
    objection: str


class Clarification(BaseModel):
    clarification: str


class QueryClass(Enum):
    USE_AVAILABLE = "use-available"
    FETCH_NEW = "fetch-new"


class QueryClassification(BaseModel):
    query_class: QueryClass
    explanation: str


class NextSteps(BaseModel):
    suggestion1: str = Field(
        description="Choose 1 - 3 available modules and present them to the user in a human, prose format and tell why they are relevant to the user's query"
    )
    suggestion2: str = Field(
        description="Choose 1 - 3 available modules and present them to the user in a human, prose format and tell why they are relevant to the user's query"
    )
    suggestion3: str = Field(
        description="Provide a general suggestion for the user to explore the data further"
    )


class XaiInsights(BaseModel):
    observations: str
    conclusions: str
    critical_reflection: str


class XaiInsights2(BaseModel):
    observations: str
    conclusions: str


class ChosenModule(BaseModel):
    module: str
    parameters: Dict[str, str] = Field(
        description="Mandatory dictionary. Leave empty, if the module needs no parameters"
    )


class ModuleChoice(BaseModel):
    module: str
    parameters: Dict[str, str] = Field(
        description="Mandatory dictionary. Provide empty dictionary, if the module needs no parameters"
    )
    explanation: str


def max_three_modules(v: List[ModuleChoice]) -> str:
    print("#################################")
    print("chosen modules:")
    print(v)
    if len(v) > 6:
        raise ValueError("The number of modules must not exceed 3")
    return v


class TrustAssessment(BaseModel):
    judgement_rating: int = Field(description="Rating for the predictions trustwortiness between 3 (Excellent), 2 (Good), 1 (Moderate), and 0 (Poor)", ge=0, le=3)
    judgement_reason: str = Field(description="A reason for the judgement rating")
    most_relevant_modules: List[str] = Field(
        min_length=1,
        max_length=2,
        description="The most relevant modules for the judgement rating (max 2)",
    )


class ModuleSummarization(BaseModel):
    summarization: str


class Modules(BaseModel):
    modules: Annotated[List[ModuleChoice], AfterValidator(max_three_modules)]


class AgentHandler:

    def __init__(
        self,
        llm,
        label_descriptions: dict = None,
        feature_descriptions: dict = None,
        module_descriptions: dict = None,
    ):
        self.df = pd.read_csv("data/full_df.csv")
        self.llm = llm
        self.label_descriptions = label_descriptions
        self.feature_descriptions = feature_descriptions
        self.module_descriptions = module_descriptions
        
        self.cache = {}

    def get_relevant_modules(self, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = initial_prompt(self.module_descriptions, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=Modules,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_initial_insights(self, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = initial_assessment_prompt(modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=XaiInsights,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_initial_insights2(self, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = initial_assessment2_prompt(modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=XaiInsights,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_insights(self, request, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = assessment_prompt(request, modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=XaiInsights,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_insights2(self, request, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = assessment_prompt2(request, modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=XaiInsights2,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def compute_next_steps(self, history, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = guidance_prompt(self.module_descriptions, datapoint, history)

        response = self.llm.generate(
            prompt,
            response_model=NextSteps,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def classify_query(self, request, modules, history, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = query_classification_prompt(
            request, modules, self.module_descriptions, history, datapoint
        )

        response = self.llm.generate(
            prompt,
            response_model=QueryClassification,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def clarify(self, request, history, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = clarification_prompt(request, history, modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=Clarification,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def objection(self, request, history, modules, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = objection_prompt(request, history, modules, datapoint)

        response = self.llm.generate(
            prompt,
            response_model=Objection,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def continuation(self, request, history, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = continuation_prompt(
            request, history, self.module_descriptions, datapoint
        )

        response = self.llm.generate(
            prompt,
            response_model=Modules,
            system_message="You are an expert in explainable AI.",
        )

        print("response:")
        print(response.dict())

        return response.dict()

    def continuation2(self, request, history, dp_id) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        prompt = continuation_prompt(
            request, history, self.module_descriptions, datapoint
        )

        response = self.llm.generate(
            prompt,
            response_model=Modules,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def module_summarization(self, module: dict, dp_id: int) -> dict:

        datapoint = retrieve_datapoint(self.df, dp_id)

        supportive_information = f"""
            Values: {datapoint["properties"]}
            Prediction: {datapoint["prediction"]["label"]}
        """

        prompt = module_summarization_prompt(json.dumps(module), supportive_information)

        response = self.llm.generate(
            prompt,
            response_model=ModuleSummarization,
            system_message="You are an expert in explainable AI.",
        )
        
        cache_key = f"{dp_id}_{module['name']}"
        
        if cache_key in self.cache:
            print(self.cache)
            return self.cache[cache_key]
        
        self.cache[cache_key] = response.dict()["summarization"]

        return response.dict()["summarization"]

    def trust_assessment(self, trace: List[Dict[str, Any]], statement: str) -> str:
        prompt = trust_assessment_prompt(trace, statement)

        response = self.llm.generate(
            prompt,
            response_model=TrustAssessment,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()

    def trust_assessment2(self, trace: List[Dict[str, Any]], statement: str) -> str:
        prompt = trust_assessment_prompt(trace, statement, sceptical=True)

        response = self.llm.generate(
            prompt,
            response_model=TrustAssessment,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
    
    def trust_assessment_with_context(self, module_insights: List[Dict[str, Any]], context: str, assessment_type: str, module_focus: str, statement: str) -> str:
        prompt = trust_assessment_with_context_prompt(module_insights, context, module_focus, statement, sceptical={assessment_type != "standard"})
        
        print(prompt)

        response = self.llm.generate(
            prompt,
            response_model=TrustAssessment,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()