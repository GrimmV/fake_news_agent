
from numbers import Number

from modules.distributions import DistributionModule
from modules.global_xai import GlobalXAIModule
from modules.individual_xai import IndividualXAIModule
from modules.performance import PerformanceModule

class ModuleCaller:
    
    def __init__(self, features: list[str]):
        self.features = features

        self.dist_module = DistributionModule(features)
        self.global_xai_module = GlobalXAIModule()
        self.individual_xai_module = IndividualXAIModule()
        self.performance_module = PerformanceModule()

    def call_module(self, module_name: str, params: dict[str, str | Number], datapoint_id: int = None):
        
        if (module_name == "feature distribution"):
            the_module = self.dist_module.get_distribution_1d(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "feature distribution 2D"):
            the_module = self.dist_module.get_distribution_2d(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "performance metrics"):
            the_module = self.performance_module.get_performances(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "confusion matrix"):
            the_module = self.performance_module.get_confusion(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "global feature importance"):
            the_module = self.global_xai_module.get_feature_importance(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "partial dependence plot"):
            print(params)
            the_module = self.global_xai_module.get_partial_dependence(**params)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "individual feature importance"):
            
            the_module = self.individual_xai_module.get_shap_values(dp_id=datapoint_id)
            visual = the_module["visual"]
            visual.update_layout(
                margin=dict(l=10, r=10, t=30, b=10)
            )
            return the_module
        elif (module_name == "similar predictions"):
            the_module = self.individual_xai_module.get_similars(dp_id=datapoint_id, **params)
            return the_module
        elif (module_name == "counterfactuals"):
            the_module = self.individual_xai_module.get_counterfactuals(dp_id=datapoint_id, **params)
            return the_module
        elif (module_name == "word importance"):
            the_module = self.individual_xai_module.get_word_shap_values(dp_id=datapoint_id, **params)
            return the_module
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