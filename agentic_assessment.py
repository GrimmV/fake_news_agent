from modules.call_module import ModuleCaller
from typing import List, Dict, Any, Callable
import json
import pandas as pd
import asyncio
from operations.utils.retrieve_datapoint import retrieve_datapoint

from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

from llm.llm import GPTModel
from operations.agent_handler import AgentHandler

import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv()


# === AGENTIC REASONING LOOP ===
async def agentic_assessment(
    predicted_label: str,
    statement: str,
    module_caller: ModuleCaller,
    agent_handler: AgentHandler,
    dp_id: int,
    websocket_send_callback: Callable[[str], None] = None,
    loop: asyncio.AbstractEventLoop = None,
) -> List[Dict[str, Any]]:
    trace = []
    
    trace, local_feature_importance_output = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="individual feature importance",
        module_params={},
        description_template="Contains the feature importance for the particular model prediction.",
        dp_id=dp_id,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )
    top_features = local_feature_importance_output["top_features"]
    feature1 = top_features[0]
    feature2 = top_features[1]

    trace, dist = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="feature distribution",
        module_params={"feature_name": feature1, "label": predicted_label},
        description_template="Contains the feature distribution for the particular model prediction.",
        dp_id=dp_id,
        feature_name_for_action=feature1,
        predicted_label_for_action=predicted_label,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )
    
    trace, dist = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="feature distribution 2D",
        module_params={"feature_name_1": feature1, "feature_name_2": feature2, "label": predicted_label},
        description_template="Contains the 2D feature distribution for the particular model prediction.",
        dp_id=dp_id,
        feature_name_for_action=[feature1, feature2],
        predicted_label_for_action=predicted_label,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )

    trace, performance_output = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="performance metrics",
        module_params={},
        description_template="Contains the performance metrics of the model.",
        dp_id=dp_id,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )
    trace, confusion_matrix_output = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="confusion matrix",
        module_params={},
        description_template="Contains the confusion matrix of the model predictions. First row/column is True, second is Neither and third is False.",
        dp_id=dp_id,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )

    trace, global_feature_importance_output = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="global feature importance",
        module_params={"label": predicted_label},
        description_template="Contains the global feature importance calculated based on the absolute sum of SHAP values.",
        dp_id=dp_id,
        predicted_label_for_action=predicted_label,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )

    trace, dist = await call_and_summarize_module(
        module_caller=module_caller,
        agent_handler=agent_handler,
        trace=trace,
        module_name="partial dependence plot",
        module_params={"feature_name": feature1, "label": predicted_label},
        description_template="Contains the partial dependence plot of {feature} towards the predicted label {predicted_label}.",
        dp_id=dp_id,
        feature_name_for_action=feature1,
        predicted_label_for_action=predicted_label,
        websocket_send_callback=websocket_send_callback,
        loop=loop,
    )

    # Step 8: Final Summary (Condensed)
    conclusion = await loop.run_in_executor(
        None,
        agent_handler.trust_assessment,
        trace,
        statement
    )
    conclusion2 = await loop.run_in_executor(
        None,
        agent_handler.trust_assessment2,
        trace,
        statement
    )
    # trace.append({"action": "final assessment", "summary": conclusion})
    if websocket_send_callback:
        payload = {
            "type": "final_assessment",
            "variant": "standard",
            "data": {"action": "final assessment", "summary": conclusion},
        }
        await websocket_send_callback(json.dumps(payload))

    if websocket_send_callback:
        payload = {
            "type": "final_assessment",
            "variant": "sceptical",
            "data": {"action": "final assessment", "summary": conclusion2},
        }
        await websocket_send_callback(json.dumps(payload))

    return trace, conclusion, conclusion2


async def call_and_summarize_module(
    module_caller,
    agent_handler,
    trace,
    module_name,
    module_params,
    description_template,
    dp_id,
    feature_name_for_action: str | List[str] = None,
    predicted_label_for_action=None,
    websocket_send_callback: Callable[[str], None] = None,
    loop: asyncio.AbstractEventLoop = None,
):
    """
    Calls a specified module, summarizes its output, appends to the trace, and sends data via WebSocket if a callback is provided.

    Args:
        module_caller: The object used to call modules.
        agent_handler: The object used for module summarization.
        trace: The list to append action and summary to.
        module_name: The name of the module to call.
        module_params: A dictionary of parameters for the module.
        description_template: A format string for the description in summarization.
                              It can use placeholders like {feature} and {predicted_label}.
        dp_id: The decision point ID.
        feature_name_for_action: (Optional) The feature name to include in the trace action.
        predicted_label_for_action: (Optional) The predicted label to include in the trace action.
        websocket_send_callback: (Optional) A callback function to send updates via WebSocket.
    """
    module_output = module_caller.call_module(
        module_name,
        module_params,
        dp_id,
    )["raw"]

    # Dynamically create the description string
    description_kwargs = {}
    if "{feature}" in description_template and "feature_name" in module_params:
        description_kwargs["feature"] = module_params["feature_name"]
    if "{predicted_label}" in description_template and "label" in module_params:
        description_kwargs["predicted_label"] = module_params["label"]

    description = description_template.format(**description_kwargs)

    summary = await loop.run_in_executor(
        None,
        agent_handler.module_summarization,
        {"name": module_name, "description": description, "output": module_output},
        dp_id,
    )

    action_description_parts = [module_name]
    if feature_name_for_action:
        if isinstance(feature_name_for_action, list):
            action_description_parts.append(f"for {', '.join(feature_name_for_action)}")
        else:
            action_description_parts.append(f"for {feature_name_for_action}")
    if predicted_label_for_action:
        action_description_parts.append(f"(label: {predicted_label_for_action})")

    action = " ".join(action_description_parts)

    trace.append({"action": action, "summary": summary, "module_name": module_name})

    if websocket_send_callback:
        payload = {
            "type": "module_update",
            "data": {
                "module": module_name,
                "params": module_params,
                "action": action,
                "summary": summary,
            },
        }
        await websocket_send_callback(json.dumps(payload))

    return trace, module_output


# === RUN EXAMPLE ===
if __name__ == "__main__":

    feature_names = [feature["name"] for feature in features]
    dp_id = 10885
    df = pd.read_csv("data/full_df.csv")
    datapoint = retrieve_datapoint(df, dp_id)
    label = datapoint["prediction"]["label"]
    API_KEY = os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    llm = GPTModel(model_name=MODEL_NAME, key=API_KEY)

    agent_handler = AgentHandler(
        llm,
        label_descriptions=labels,
        feature_descriptions=features,
        module_descriptions=module_descriptions,
    )
    module_caller = ModuleCaller([feature["name"] for feature in features])
    print(label)
    result = agentic_assessment(
        predicted_label=label,
        module_caller=module_caller,
        agent_handler=agent_handler,
        dp_id=dp_id,
        websocket_send_callback=None,
    )
    for step in result:
        print(f"\n[Action: {step['action']}]")
        print(json.dumps(step["summary"], indent=2))
