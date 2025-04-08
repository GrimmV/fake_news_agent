
import asyncio
import websockets
import json

from config import datapoint_id

from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

from modules.call_module import ModuleCaller

from llm.llm import GPTModel
from operations.agent_handler import AgentHandler

import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv()

API_KEY = os.getenv('API_KEY')
MODEL_NAME = os.getenv("MODEL_NAME")
llm = GPTModel(model_name=MODEL_NAME, key=API_KEY)

agent_handler = AgentHandler(llm, label_descriptions=labels, feature_descriptions=features, module_descriptions=module_descriptions)
module_caller = ModuleCaller([feature["name"] for feature in features])

async def workflow(websocket):
    
    loop = asyncio.get_event_loop()
    message_id = 0
    history = []

    async for message in websocket:
        info = json.loads(message)
        request = info["request"]
        rq_type = info["type"]
        
        print(message)
        
        if rq_type == "initial":
            
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Choosing Relevant Data..."}))
            modules = await loop.run_in_executor(
                None, agent_handler.get_relevant_modules, datapoint_id
            )
            history.append(f"Modules chosen by the assistant: {modules}")
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": "Relevant Data Chosen"}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "modules", "status": "done", "content": modules}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Retrieve Data..."}))
            modules_data = await loop.run_in_executor(
                None, module_caller.collect_data, modules["modules"], datapoint_id
            )
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": "Data Retrieved"}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Derive Insights..."}))
            initial_insights = await loop.run_in_executor(
                None, agent_handler.compute_insights, modules_data, datapoint_id
            )
            history.append(f"Based on the observed data, the Assistant concluded: {initial_insights['conclusions']}")
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": "Insights Derived"}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "insights", "status": "done", "content": initial_insights}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Thinking about Next Steps..."}))
            next_steps = await loop.run_in_executor(
                None, agent_handler.compute_next_steps, history, datapoint_id
            )
            history.append(f"The assistant suggested the following next steps: {next_steps}")
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": "Collected Next Steps"}))
            
            message_id += 1
            await websocket.send(json.dumps({"id": message_id, "type": "next_steps", "status": "done", "content": next_steps}))
            
            message_id += 1
        
        elif rq_type == "request":
                        
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Classifying User Query..."}))
            query_classification = await loop.run_in_executor(
                None, agent_handler.classify_query, request, history, datapoint_id
            )
            query_class = query_classification['query_class'].value
            query_class_explanation = query_classification['explanation']
            await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": f"Classified User Query as: {query_class}"}))
            
            message_id += 1
            if query_class == "out-of-scope":
                response = f'''
                    Your request has been classified as out-of-scope because: {query_class_explanation}\\
                    Please make a request inside of the applications scope.
                '''
                await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": response}))
            elif query_class == "ambiguous":
                response = f'''
                    Your request has been classified as ambiguous because: {query_class_explanation}\\
                    Please clarify your request so that the assistant can respond accordingly.
                '''
                await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": response}))
            elif query_class == "other":
                response = f'''
                    Your request has been classified as "other" because: {query_class_explanation}\\
                    Please write a new request so that it can be properly classified in the application scope.
                '''
                await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": response}))
            elif query_class == "clarification":
                await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "pending", "content": "Thinking about clarification..."}))
                clarification = await loop.run_in_executor(
                    None, agent_handler.clarify, request, history, modules, datapoint_id
                )
                await websocket.send(json.dumps({"id": message_id, "type": "processing", "status": "done", "content": "Finished clarification"}))
                
                message_id += 1
                await websocket.send(json.dumps({"id": message_id, "type": "clarification", "status": "done", "content": clarification["clarification"]}))
                
            

            

async def main():
    async with websockets.serve(workflow, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())