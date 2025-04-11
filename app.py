from flask import Flask, request, make_response
import pandas as pd

from cors_handling import _corsify_actual_response, _build_cors_preflight_response

from config import base_url

from operations.utils.retrieve_datapoint import retrieve_datapoint
from descriptions.features import features
from modules.call_module import ModuleCaller

app = Flask(__name__)
module_caller = ModuleCaller([feature["name"] for feature in features])

df = pd.read_csv("data/full_df.csv")

@app.route(f"{base_url}/visual", methods=["POST", "OPTIONS"])
def get_visual():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        request_object = request.get_json()
        module = request_object["module"]
        params = request_object["params"]
        datapoint_id = request_object["datapoint_id"]
        username = request_object["username"]
        the_module = module_caller.call_module(module_name=module, params=params, datapoint_id=datapoint_id)
        visual = the_module["visual"]
        raw = the_module["raw"]
        if visual != None:
            html_visual = visual.to_html()
            response = make_response(html_visual)
        else:
            response = make_response(raw)
        return _corsify_actual_response(response)
    
@app.route(f"{base_url}/prediction", methods=["GET", "OPTIONS"])
def get_prediction():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "GET":
        username = request.args.get('username')
        datapoint_id = request.args.get('datapoint_id')
        datapoint_id = int(datapoint_id)
        prediction = retrieve_datapoint(df, datapoint_id)
        response = make_response(prediction)
        return _corsify_actual_response(response)

