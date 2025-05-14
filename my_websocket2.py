import asyncio
import websockets
import json

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
    modules = {}

    async for message in websocket:
        info = json.loads(message)
        request = info["request"]
        rq_type = info["type"]
        username = info["username"]
        datapoint_id = info["datapoint_id"]
        
        try:
            histories
        except NameError:
            histories = {}

        if username not in usernames:
            handle_unallowed_username(websocket, 0, username)
            continue
        if username not in histories:
            histories[username] = {datapoint_id: []}
            message_ids[username] = {datapoint_id: 0}
        elif datapoint_id not in histories[username]:
            histories[username][datapoint_id] = []
            message_ids[username][datapoint_id] = 0

        message_id = message_ids[username][datapoint_id]
        history = histories[username][datapoint_id]
        

        if rq_type == "initial":

            history, message_id = await compute_next_steps(websocket, message_id, loop, history, datapoint_id, username)
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
                None,
                agent_handler.classify_query,
                request,
                modules["modules"] if "modules" in modules else [],
                history[-history_lookback:],
                datapoint_id,
            )
            query_class = query_classification["query_class"].value
            query_class_explanation = query_classification["explanation"]
            print(query_class)
            print(query_class_explanation)
            tmp_message = "Fetching New Data" if query_class == "fetch-new" else "Handling User Request"
            print(tmp_message)
            await websocket.send(
                json.dumps(
                    {
                        "id": message_id,
                        "type": "processing",
                        "status": "done",
                        "content": tmp_message,
                    }
                )
            )

            message_id = increment_message_id(message_ids, username, datapoint_id)
            # if query_class == "fetch-new":
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
                None,
                agent_handler.continuation,
                request,
                history[-history_lookback:],
                datapoint_id,
            )
            history.append(f"User request: {request}")
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
            history, message_id = await module_assessment(
                history,
                modules,
                message_id,
                websocket,
                loop,
                username,
                datapoint_id,
                message_ids,
                request,
                insight_type="request"
            )
            # else:
            #     history, message_id = await module_assessment(
            #         history,
            #         modules,
            #         message_id,
            #         websocket,
            #         loop,
            #         username,
            #         datapoint_id,
            #         message_ids,
            #         request,
            #         insight_type="request"
            #     )


def add_parameter_options(modules):
    
    if "modules" not in modules:
        return {"modules": []}

    new_modules = {"modules": []}

    print("#########################################################")
    print(modules)
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
        break
    

    print(new_modules)

    return new_modules


async def module_assessment(
    history,
    modules,
    message_id,
    websocket,
    loop,
    username,
    datapoint_id,
    message_ids,
    request="",
    insight_type="initial",
):

    print("#########################################################")
    print(modules)

    modules = add_parameter_options(modules)
    await websocket.send(
        json.dumps(
            {"id": message_id, "type": "modules", "status": "done", "content": modules}
        )
    )

    message_id = increment_message_id(message_ids, username, datapoint_id)

    message_id, modules_data = await retrieving_data(
        websocket, message_id, loop, modules, username, datapoint_id
    )

    message_id, history = await compute_insights(
        websocket,
        message_id,
        loop,
        modules_data,
        history,
        username,
        datapoint_id,
        request,
        insight_type,
    )

    history, message_id = await compute_next_steps(
        websocket, message_id, loop, history, datapoint_id, username
    )

    return history, message_id


async def retrieving_data(websocket, message_id, loop, modules, username, datapoint_id):
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

    return message_id, modules_data


async def compute_insights(
    websocket,
    message_id,
    loop,
    modules_data,
    history,
    username, datapoint_id,
    request="",
    insight_type="initial",
):
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
    if insight_type == "initial":
        insights = await loop.run_in_executor(
            None, agent_handler.compute_initial_insights2, modules_data, datapoint_id
        )
    else:
        insights = await loop.run_in_executor(
            None, agent_handler.compute_insights2, request, modules_data, datapoint_id
        )

    history.append(
        f"Based on the observed data, the Assistant concluded: {insights['conclusions']}"
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
                "content": insights,
            }
        )
    )
    message_id = increment_message_id(message_ids, username, datapoint_id)

    return message_id, history


async def compute_next_steps(websocket, message_id, loop, history, datapoint_id, username):
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
        None,
        agent_handler.compute_next_steps,
        history[-history_lookback:],
        datapoint_id,
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

    return history, message_id


def increment_message_id(message_ids, username, datapoint_id):
    message_ids[username][datapoint_id] += 1
    return message_ids[username][datapoint_id]


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


async def identify_initial_modules(websocket, message_id, loop, datapoint_id, username):
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

    return modules, message_id


async def main():
    async with websockets.serve(workflow, "localhost", 8766):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
