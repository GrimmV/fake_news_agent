from flask import Flask, request, make_response
from numbers import Number

from cors_handling import _corsify_actual_response, _build_cors_preflight_response

from config import base_url, datapoint_id

from modules.distributions import DistributionModule
from modules.global_xai import GlobalXAIModule
from modules.individual_xai import IndividualXAIModule
from modules.performance import PerformanceModule

app = Flask(__name__)

dist_module = DistributionModule()
global_xai_module = GlobalXAIModule()
individual_xai_module = IndividualXAIModule()
performance_module = PerformanceModule()

@app.route(f"{base_url}/visual", methods=["POST", "OPTIONS"])
def get_visual():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        request_object = request.get_json()
        module = request_object["module"]
        params = request_object["params"]
        visual = _call_module(module_name=module, params=params)
        response = make_response(visual)
        return _corsify_actual_response(response)
    
def _call_module(module_name: str, params: dict[str, str | Number]):
    
    if (module_name == "feature distribution"):
        the_module = dist_module.get_distribution_1d(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "feature distribution 2D"):
        the_module = dist_module.get_distribution_2d(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "performance metrics"):
        the_module = performance_module.get_performances(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "confusion matrix"):
        the_module = performance_module.get_confusion(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "global feature importance"):
        the_module = global_xai_module.get_feature_importance(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="LightSteelBlue",
        )
        return visual.to_html()
    if (module_name == "partial dependence plot"):
        the_module = global_xai_module.get_partial_dependence(**params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "individual feature importance"):
        the_module = individual_xai_module.get_shap_values(dp_id=datapoint_id, **params)
        visual = the_module["visual"]
        visual.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return visual.to_html()
    if (module_name == "similar predictions"):
        the_module = individual_xai_module.get_similars(**params)
        raw = the_module["raw"]
        return raw
    if (module_name == "counterfactuals"):
        the_module = individual_xai_module.get_counterfactuals(**params)
        raw = the_module["raw"]
        return raw
    else:
        return ""
    
