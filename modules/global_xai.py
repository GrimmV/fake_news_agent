import json
import plotly.express as px

from modules.utils.label_mapper import label_map

class GlobalXAIModule:
    
    def __init__(self):
        feature_imp_location = "data/feature_importance.csv"
        pdp_location = "data/pdp.csv"
        
        with open(feature_imp_location) as f:
            self.feature_importances = json.load(f)
        with open(pdp_location) as f:
            self.pdp = json.load(f)
            
    def get_feature_importance(self, label):
        
        my_label = label_map[label]
        
        features = self.feature_importances[label].keys()
        values = self.feature_importances[label].values()
        
        raw = {
            "label": my_label,
            "values": self.feature_importances[label]
        }
        
        visual = px.bar(x=features, y=values)
        
        return {
            "raw": raw,
            "visual": visual        
        }
    
    def get_partial_dependence(self, feature_name, class_label):
        
        my_label = label_map[class_label]
        
        elem = list(filter(lambda x: x["feature"] == feature_name, self.pdp))[0]
        grid_values = elem["partial_dependence"]["grid_values"][0]
        average = elem["partial_dependence"]["average"][my_label]

        raw = {
            "feature": feature_name,
            "class": class_label,
            "grid_values": grid_values,
            "average": average
        }
        
        visual = px.line(x=grid_values, y=average)
        
        return {
            "raw": raw,
            "visual": visual
        }
        
if __name__ == "__main__":
    
    module = GlobalXAIModule()
    pdp = module.get_partial_dependence("Lexical Diversity (TTR)", "False")
    
    pdp["visual"].show()
    # print(module.get_feature_importances())
        