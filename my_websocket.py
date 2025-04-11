import asyncio
import websockets
import json

from config import datapoint_id

from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

from login.usernames import usernames

from modules.call_module import ModuleCaller

from llm.llm import GPTModel
from operations.agent_handler import AgentHandler

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

history_lookback = 3
message_ids = {}
histories = {}

async def workflow(websocket):

    loop = asyncio.get_event_loop()

    async for message in websocket:
        info = json.loads(message)
        request = info["request"]
        rq_type = info["type"]
        username = info["username"]
        datapoint_id = info["datapoint_id"]

        if username not in usernames:
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
            continue
        if username not in histories:
            histories[username] = {
                datapoint_id: []
            }
            message_ids[username] = {
                datapoint_id: 0
            }
        elif datapoint_id not in histories[username]:
            histories[username][datapoint_id] = []
            message_ids[username][datapoint_id] = 0
            
        message_id = message_ids[username][datapoint_id]
        history = histories[username][datapoint_id]
        
        print(histories)
        print(message_ids)
        
        if rq_type == "initial":
            
            await websocket.send(
                json.dumps(
                    {
                        "id": message_id,
                        "type": "processing",
                        "status": "pending",
                        "content": "Choosing Relevant Data...",
                    }
                )
            )
            modules = await loop.run_in_executor(
                None, agent_handler.get_relevant_modules, datapoint_id
            )
            history, message_id, modules_data = await module_assessment(
                history, modules, message_id, websocket, loop, username, datapoint_id, message_ids
            )
        elif rq_type == "request":

            # +1 for the new user query
            message_id = increment_message_id(message_ids, username, datapoint_id)

            await websocket.send(
                json.dumps(
                    {
                        "id": message_id,
                        "type": "processing",
                        "status": "pending",
                        "content": "Classifying User Query...",
                    }
                )
            )
            query_classification = await loop.run_in_executor(
                None, agent_handler.classify_query, request, history[-history_lookback:], datapoint_id
            )
            query_class = query_classification["query_class"].value
            query_class_explanation = query_classification["explanation"]
            await websocket.send(
                json.dumps(
                    {
                        "id": message_id,
                        "type": "processing",
                        "status": "done",
                        "content": f"Classified User Query as: {query_class}",
                    }
                )
            )

            message_id = increment_message_id(message_ids, username, datapoint_id)
            if query_class == "out-of-scope":
                response = f"""
                    Your request has been classified as out-of-scope because: {query_class_explanation}\\
                    Please make a request inside of the applications scope.
                """
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "done",
                            "content": response,
                        }
                    )
                )
                message_id = increment_message_id(message_ids, username, datapoint_id)
            elif query_class == "ambiguous":
                history.append(f"User request: {request}")
                response = f"""
                    Your request has been classified as ambiguous because: {query_class_explanation}\\
                    Please clarify your request so that the assistant can respond accordingly.
                """
                history.append(
                    f"Assistant handled ambiguity: {query_class_explanation}"
                )
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "done",
                            "content": response,
                        }
                    )
                )
                message_id = increment_message_id(message_ids, username, datapoint_id)
            elif query_class == "clarification":
                history.append(f"User request: {request}")
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "pending",
                            "content": "Thinking about clarification...",
                        }
                    )
                )
                clarification = await loop.run_in_executor(
                    None,
                    agent_handler.clarify,
                    request,
                    history[-history_lookback:],
                    modules_data,
                    datapoint_id,
                )
                history.append(f"Assistant offered clarification: {clarification}")
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "done",
                            "content": "Finished clarification",
                        }
                    )
                )

                message_id = increment_message_id(message_ids, username, datapoint_id)
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "clarification",
                            "status": "done",
                            "content": clarification["clarification"],
                        }
                    )
                )
                message_id = increment_message_id(message_ids, username, datapoint_id)
            elif query_class == "objection":
                history.append(f"User request: {request}")
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "pending",
                            "content": "Assessing objection...",
                        }
                    )
                )
                objection = await loop.run_in_executor(
                    None,
                    agent_handler.objection,
                    request,
                    history[-history_lookback:],
                    modules_data,
                    datapoint_id,
                )
                history.append(f"Assistant handled user objection: {objection}")
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "done",
                            "content": "Finished assessment",
                        }
                    )
                )

                message_id = increment_message_id(message_ids, username, datapoint_id)
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "objection",
                            "status": "done",
                            "content": objection["objection"],
                        }
                    )
                )
                message_id = increment_message_id(message_ids, username, datapoint_id)
            elif query_class == "continuation":
                history.append(f"User request: {request}")
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "pending",
                            "content": "Choosing Relevant Data...",
                        }
                    )
                )
                modules = await loop.run_in_executor(
                    None, agent_handler.continuation, request, history[-history_lookback:], datapoint_id
                )
                history, message_id, modules_data = await module_assessment(
                    history, modules, message_id, websocket, loop, username, datapoint_id, message_ids
                )
            else:
                response = f"""
                    Your request has been classified as "other" because: {query_class_explanation}\\
                    Please write a new request so that it can be properly classified in the application scope.
                """
                await websocket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "type": "processing",
                            "status": "done",
                            "content": response,
                        }
                    )
                )
                message_id = increment_message_id(message_ids, username, datapoint_id)


def add_parameter_options(modules):

    new_modules = {"modules": []}

    my_modules = modules["modules"]

    for module in my_modules:
        param_options = {}
        name = module["module"]
        elem = list(filter(lambda x: x["name"] == name, module_descriptions))[0]
        params = elem["parameters"]
        for param in params:
            if "feature_name" in param:
                param_options[param] = [feature["name"] for feature in features]
            if "label" in param:
                param_options[param] = ["True", "Neither", "False"]
        module["param_options"] = param_options
        new_modules["modules"].append(module)

    return new_modules


async def module_assessment(history, modules, message_id, websocket, loop, username, datapoint_id, message_ids):

    modules = add_parameter_options(modules)

    history.append(f"Modules chosen by the assistant: {modules['modules']}")
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "done",
                "content": "Relevant Data Chosen",
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {"id": message_id, "type": "modules", "status": "done", "content": modules}
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "pending",
                "content": "Retrieve Data...",
            }
        )
    )
    modules_data = await loop.run_in_executor(
        None, module_caller.collect_data, modules["modules"], datapoint_id
    )
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "done",
                "content": "Data Retrieved",
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "pending",
                "content": "Derive Insights...",
            }
        )
    )
    initial_insights = await loop.run_in_executor(
        None, agent_handler.compute_insights, modules_data, datapoint_id
    )
    history.append(
        f"Based on the observed data, the Assistant concluded: {initial_insights['conclusions']}"
    )
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "done",
                "content": "Insights Derived",
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "insights",
                "status": "done",
                "content": initial_insights,
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "pending",
                "content": "Thinking about Next Steps...",
            }
        )
    )
    next_steps = await loop.run_in_executor(
        None, agent_handler.compute_next_steps, history[-history_lookback:], datapoint_id
    )
    history.append(f"The assistant suggested the following next steps: {next_steps}")
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "processing",
                "status": "done",
                "content": "Collected Next Steps",
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)
    await websocket.send(
        json.dumps(
            {
                "id": message_id,
                "type": "next_steps",
                "status": "done",
                "content": next_steps,
            }
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)

    return history, message_id, modules_data

def increment_message_id(message_ids, username, datapoint_id):
    message_ids[username][datapoint_id] += 1
    return message_ids[username][datapoint_id]

async def main():
    async with websockets.serve(workflow, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
