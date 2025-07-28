import pandas as pd
from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions
import asyncio
from modules.call_module import ModuleCaller

from llm.llm import GPTModel
from operations.agent_handler import AgentHandler
from agentic_assessment_sync import agentic_assessment
from operations.utils.retrieve_datapoint import retrieve_datapoint
import os
from dotenv import load_dotenv
from init_phoenix import init_phoenix
from opentelemetry.trace import StatusCode

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME_2 = os.getenv("MODEL_NAME_2")
llm = GPTModel(model_name=MODEL_NAME, key=API_KEY)
llm_2 = GPTModel(model_name=MODEL_NAME_2, key=API_KEY)

tracer = init_phoenix("experiment_workflow_test_0.1")

agent_handler = AgentHandler(
    llm,
    label_descriptions=labels,
    feature_descriptions=features,
    module_descriptions=module_descriptions,
    tracer=tracer
)
agent_handler_2 = AgentHandler(
    llm_2,
    label_descriptions=labels,
    feature_descriptions=features,
    module_descriptions=module_descriptions,
    tracer=tracer
)
module_caller = ModuleCaller([feature["name"] for feature in features], tracer)

df = pd.read_csv("data/full_df.csv")

def experiment_workflow(datapoint_id):
    
    if isinstance(datapoint_id, dict):
        datapoint_id = datapoint_id["dp_id"]
    
    datapoint = retrieve_datapoint(df, datapoint_id)
    label = datapoint["prediction"]["label"]
    statement = datapoint["statement"]
    date = datapoint["date"]
    author = datapoint["author"]
        
    combined_statement = f"{author} ({date}): {statement}"

    return agentic_assessment(
        predicted_label=label,
        statement=combined_statement,
        module_caller=module_caller,
        agent_handler=agent_handler,
        agent_handler_2=agent_handler_2,
        dp_id=datapoint_id,
    )

if __name__ == "__main__":
    with tracer.start_as_current_span(
        "ExperimentWorkflow", openinference_span_kind="agent"
    ) as span:
        experiment_workflow(220)
        span.set_status(StatusCode.OK)