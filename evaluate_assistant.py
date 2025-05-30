import json
import os
import asyncio

import pandas as pd

from operations.agent_handler import AgentHandler
from operations.utils.retrieve_datapoint import retrieve_datapoint
from agentic_assessment import agentic_assessment
from llm.llm import GPTModel
from modules.call_module import ModuleCaller
from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

async def main():
    API_KEY = os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    llm = GPTModel(model_name=MODEL_NAME, key=API_KEY)
    loop = asyncio.get_event_loop()
    agent_handler = AgentHandler(
        llm,
        label_descriptions=labels,
        feature_descriptions=features,
        module_descriptions=module_descriptions,
    )
    module_caller = ModuleCaller([feature["name"] for feature in features])
    
    df = pd.read_csv("data/full_df.csv")
    results = []
    for i, row in df.iterrows():
        print(i)
        dp_id = row["id"]
        dp_label = row["new_labels"]
        datapoint = retrieve_datapoint(df, dp_id)
        label = datapoint["prediction"]["label"]
        if label == "False" and dp_label != 0:
            trace, conclusion1, conclusion2 = await agentic_assessment(
                predicted_label=label,
                module_caller=module_caller,
                agent_handler=agent_handler,
                dp_id=dp_id,
                websocket_send_callback=None,
                loop=loop,
            )
            results.append({
                "statement": row["statement"],
                "dp_id": dp_id,
                "dp_label": dp_label,
                "label": label,
                "assessment1": conclusion1["trustworthiness"],
                "assessment2": conclusion2["trustworthiness"],
            })
            if len(results) % 5 == 0 and len(results) != 0:
                pd.DataFrame(results).to_csv(f"assistant_evaluation/results_checkpoint_wrong_pred{i}.csv", index=False)
        if len(results) >= 50:
            break
    for i, row in df.iterrows():
        print(i)
        dp_id = row["id"]
        dp_label = row["new_labels"]
        datapoint = retrieve_datapoint(df, dp_id)
        label = datapoint["prediction"]["label"]
        if label == "False" and dp_label == 0:
            trace, conclusion1, conclusion2 = await agentic_assessment(
                predicted_label=label,
                module_caller=module_caller,
                agent_handler=agent_handler,
                dp_id=dp_id,
                websocket_send_callback=None,
                loop=loop,
            )
            results.append({
                "statement": row["statement"],
                "dp_id": dp_id,
                "dp_label": dp_label,
                "label": label,
                "assessment1": conclusion1["trustworthiness"],
                "assessment2": conclusion2["trustworthiness"],
            })
            if len(results) % 5 == 0 and len(results) != 0:
                pd.DataFrame(results).to_csv(f"assistant_evaluation/results_checkpoint_true_pred{i}.csv", index=False)
        if len(results) >= 100:
            break
    print(results)
    # save results to csv
    pd.DataFrame(results).to_csv("assistant_evaluation/results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
