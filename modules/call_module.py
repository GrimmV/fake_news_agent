from modules.distributions import DistributionModule
from modules.global_xai import GlobalXAIModule
from modules.individual_xai import IndividualXAIModule
from modules.performance import PerformanceModule
from modules.utils.return_module import return_module

class ModuleCaller:
    
    def __init__(self, features: list[str], tracer):
        self.features = features
        self.tracer = tracer

        self.dist_module = DistributionModule(features)
        self.global_xai_module = GlobalXAIModule()
        self.individual_xai_module = IndividualXAIModule()
        self.performance_module = PerformanceModule()

    def call_module(self, module_name: str, params: dict[str, str], datapoint_id: int = None):
        
        with self.tracer.start_as_current_span(module_name, openinference_span_kind="chain") as span:
            if (module_name == "feature distribution"):
                return return_module(self.dist_module.get_distribution_1d, params, span=span)
            elif (module_name == "feature distribution 2D"):
                return return_module(self.dist_module.get_distribution_2d, params, span=span)
            elif (module_name == "performance metrics"):
                return return_module(self.performance_module.get_performances, params, span=span)
            elif (module_name == "confusion matrix"):
                return return_module(self.performance_module.get_confusion, params, span=span)
            elif (module_name == "global feature importance"):
                return return_module(self.global_xai_module.get_feature_importance, params, span=span)
            elif (module_name == "partial dependence plot"):
                return return_module(self.global_xai_module.get_partial_dependence, params, span=span)
            elif (module_name == "individual feature importance"):
                return return_module(self.individual_xai_module.get_shap_values, params={}, datapoint_id=datapoint_id, span=span)
            else:
                return None
        
    def collect_data(self, modules, datapoint_id):
        modules_data = []
        
        for my_module in modules:
            module = my_module["module"]
            params = my_module["parameters"]
            the_module = self.call_module(module_name=module, params=params, datapoint_id=datapoint_id)
            raw_data = the_module["raw"]
            if raw_data != None:
                modules_data.append({
                    "name": module,
                    "params": params,
                    "data": raw_data
                })
                
        return modules_data