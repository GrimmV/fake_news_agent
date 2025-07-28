import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing_extensions import Annotated
from pydantic import AfterValidator
from enum import Enum

from prompt_templates.module_summarization import module_summarization_prompt
from prompt_templates.trust_assessment import trust_assessment_prompt
from prompt_templates.trust_assessment_with_context import (
    trust_assessment_with_context_prompt,
)
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
    judgement_rating: int = Field(
        description="Rating for the predictions trustwortiness between 3 (Excellent), 2 (Good), 1 (Moderate), and 0 (Poor)",
        ge=0,
        le=3,
    )
    judgement_reason: str = Field(description="A reason for the judgement rating")
    judgement_reason_short: str = Field(
        description="One sentence summary of the judgement reason in a way that is understandable by a 5th grader"
    )
    most_relevant_modules: List[str] = Field(
        min_length=1,
        max_length=2,
        description="The most relevant modules for the judgement rating (max 2)",
    )
    # most_likely_class: str = Field(
    #     description="Given your judgement rating and the statement, if you had to decide on the most likely class yourself (True, Neither or False), which class would you choose? Provide a short explanation."
    # )


class ModuleSummarization(BaseModel):
    summarization: str
    laymans_summary: str = Field(
        description="A laymans summary of the summarization being understandable by a 5th grader"
    )


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

        self.cache[cache_key] = response.dict()

        return response.dict()

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

    def trust_assessment_with_context(
        self,
        module_insights: List[Dict[str, Any]],
        context: str,
        assessment_type: str,
        module_focus: str,
        statement: str,
    ) -> str:
        prompt = trust_assessment_with_context_prompt(
            module_insights,
            context,
            module_focus,
            statement,
            sceptical={assessment_type != "standard"},
        )

        print(prompt)

        response = self.llm.generate(
            prompt,
            response_model=TrustAssessment,
            system_message="You are an expert in explainable AI.",
        )

        return response.dict()
