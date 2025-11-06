from flask import Flask, request, make_response, jsonify
import pandas as pd
import json
import os
from glob import glob
import random
from cors_handling import _corsify_actual_response, _build_cors_preflight_response
from config import base_url
from operations.utils.retrieve_datapoint import retrieve_datapoint
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv(override=True)

import instructor

app = Flask(__name__)

# Load the dataframe
df = pd.read_csv("data/full_df.csv")
datapoint_ids = [30, 31, 32, 33, 34, 35, 65, 66, 67, 68, 69, 70, 197, 198, 199, 200, 201, 202, 255, 256, 257, 258, 259, 260]
api_key = os.getenv("API_KEY")
client = instructor.from_openai(OpenAI(api_key=api_key))

class BaseResponseModel(BaseModel):
    response: str

@app.route(f"{base_url}/posts", methods=["GET", "OPTIONS"])
def get_posts():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "GET":
        try:
            
            posts = []
            
            for dp_id in datapoint_ids:
                try:
                    # Retrieve datapoint information
                    datapoint = retrieve_datapoint(df, dp_id)
                    
                    likes = random.randint(0, 1000);
                    comments = random.randint(0, likes);
                    shares = random.randint(0, likes);
                    # Map the datapoint to the required post format
                    post = {
                        "id": dp_id,
                        "name": datapoint["author"],
                        "profileImage": datapoint["avatar"],
                        "date": datapoint["date"],
                        "content": datapoint["statement"],
                        "likes": likes,
                        "comments": comments,
                        "shares": shares,
                        "isFakeNews": datapoint["prediction"]["label"] == "False",
                        "features": datapoint["properties"]
                    }
                    
                    posts.append(post)
                    
                except Exception as e:
                    # Skip invalid datapoint IDs and continue
                    print(f"Error processing datapoint {dp_id}: {str(e)}")
                    continue
            
            response = make_response(jsonify({"posts": posts}))
            return _corsify_actual_response(response)
            
        except Exception as e:
            error_response = make_response(jsonify({"error": f"Internal server error: {str(e)}"}), 500)
            return _corsify_actual_response(error_response)

@app.route(f"{base_url}/evaluation_data", methods=["POST", "OPTIONS"])
def get_evaluation_data():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        try:
            request_object = request.get_json()
            print(request_object)
            datapoint_ids_local = request_object.get("datapoint_ids")
            
            if not datapoint_ids_local:
                return _corsify_actual_response(make_response(jsonify({"error": "No datapoint IDs provided"}), 400))
            
            evaluation_data = []
            
            # Search through all subdirectories in experiments_raw_v1.2
            base_path = "observations/experiments_raw_v1.2"
            subdirs = ["false", "mostly_true", "pof", "true"]
            
            for dp_id in datapoint_ids_local:
                found = False
                
                # Search through all subdirectories
                for subdir in subdirs:
                    if found:
                        break
                        
                    # Get all exp-*.json files in this subdirectory
                    exp_files = glob(f"{base_path}/{subdir}/exp-1.json")
                    
                    for exp_file in exp_files:
                        try:
                            with open(exp_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Search for the datapoint in this file
                            for entry in data:
                                if entry.get("input", {}).get("dp_id") == dp_id:
                                    
                                    visualizations = []
                                    
                                    modules = entry.get("output", {}).get("trace", [])
                                    for module in modules:
                                        visualization = {
                                            "title": module.get("module_name"),
                                            "description": module.get("laymans_summary"),
                                            "extended_description": module.get("summary"),
                                            "data": module.get("module_output"),
                                        }
                                        visualizations.append(visualization)
                                    conclusion = entry.get("output", {}).get("conclusion", {})
                                    
                                    evaluation_entry = {
                                        "id": dp_id,
                                        "trustScore": "Low" if conclusion.get("judgement_rating") <= 1 else "High",
                                        "rationale": conclusion.get("judgement_reason_short"),
                                        "detailedAnalysis": conclusion.get("judgement_reason"),
                                        "visualizations": {
                                            "id": dp_id,
                                            "visualizations": visualizations
                                        }
                                    }
                                    
                                    evaluation_data.append(evaluation_entry)
                                    found = True
                                    break
                                    
                        except Exception as e:
                            print(f"Error reading file {exp_file}: {str(e)}")
                            continue
                
                if not found:
                    print(f"Datapoint {dp_id} not found in any experiment files")
                    # Add empty entry for missing datapoint
                    evaluation_data.append({
                        "id": dp_id,
                        "trustScore": None,
                        "rationale": "Data not found",
                        "detailedAnalysis": "This datapoint was not found in the experiment files"
                    })
            
            response = make_response(jsonify({"evaluation_data": evaluation_data}))
            return _corsify_actual_response(response)
            
        except Exception as e:
            error_response = make_response(jsonify({"error": f"Internal server error: {str(e)}"}), 500)
            return _corsify_actual_response(error_response)


@app.route(f"{base_url}/chat", methods=["POST", "OPTIONS"])
def chat_completion():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        try:
            body = request.get_json() or {}
            prompt = body.get("prompt")
            model = "gpt-4.1-mini"
            
            print(prompt)

            if not prompt:
                return _corsify_actual_response(make_response(jsonify({"error": "Missing 'prompt' in request body"}), 400))

            if not api_key:
                return _corsify_actual_response(make_response(jsonify({"error": "API_KEY not set in environment"}), 500))


            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant that addresses the user's request in a to the point manner.",
                }, {
                    "role": "user", "content": prompt
                }],
                response_model=BaseResponseModel,
            )
            
            print(completion.model_dump())

            message_content = completion.response

            response = make_response(
                jsonify({
                    "response": message_content,
                })
            )
            return _corsify_actual_response(response)

        except Exception as e:
            error_response = make_response(jsonify({"error": f"Internal server error: {str(e)}"}), 500)
            return _corsify_actual_response(error_response)


if __name__ == "__main__":
    app.run(debug=True)
