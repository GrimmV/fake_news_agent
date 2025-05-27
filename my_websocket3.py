import asyncio
import websockets
import json
import pandas as pd
from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

from login.usernames import usernames

from modules.call_module import ModuleCaller

from llm.llm import GPTModel
from operations.agent_handler import AgentHandler
from agentic_assessment import agentic_assessment
from operations.utils.retrieve_datapoint import retrieve_datapoint
import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv()

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

df = pd.read_csv("data/full_df.csv")


async def workflow(websocket):

    loop = asyncio.get_event_loop()

    async for message in websocket:
        info = json.loads(message)
        username = info["username"]
        datapoint_id = info["datapoint_id"]
        datapoint = retrieve_datapoint(df, datapoint_id)
        label = datapoint["prediction"]["label"]

        # if username not in usernames:
        #     handle_unallowed_username(websocket, 0, username)
        #     continue
        if info["type"] == "initialization":
            await agentic_assessment(
                predicted_label=label,
                module_caller=module_caller,
                agent_handler=agent_handler,
                dp_id=datapoint_id,
                websocket_send_callback=websocket.send,
                loop=loop,
            )
        elif info["type"] == "update_assessment":
            meta_info = info["meta"]
            context = meta_info["context"]
            assessment_type = meta_info["assessment_type"]
            module_focus = meta_info["module_focus"]
            module_insights = meta_info["modules"]
            conclusion = await loop.run_in_executor(
                None,
                agent_handler.trust_assessment_with_context,
                module_insights,
                context,
                assessment_type,
                module_focus,
            )
            payload = {
                "type": "final_assessment",
                "variant": assessment_type,
                "data": {"action": "final assessment", "summary": conclusion},
            }
            print(payload)
            await websocket.send(json.dumps(payload))


async def handle_unallowed_username(websocket, message_id, username):

    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "done",
                "content": f"You, {username}, are not eligible to interact with the assistant",
            }
        )
    )


async def main():
    async with websockets.serve(workflow, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
