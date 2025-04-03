import json
import plotly.express as px

class GlobalXAIModule:
    
    def __init__(self):
        feature_imp_location = "data/feature_importance.csv"
        pdp_location = "data/pdp.csv"
        
        with open(feature_imp_location) as f:
            self.feature_importances = json.load(f)
        with open(pdp_location) as f:
            self.pdp = json.load(f)
            
    def get_feature_importance(self, label):
        
        label_key = f"class_{label}"
        
        features = self.feature_importances[label_key].keys()
        values = self.feature_importances[label_key].values()
        
        raw = {
            "label": label,
            "values": self.feature_importances[label_key]
        }
        
        visual = px.bar(x=features, y=values)
        
        return {
            "raw": raw,
            "visual": visual        
        }
    
    def get_partial_dependence(self, feature_name, class_label):
        
        label_mapper = {
            "True": 0,
            "Neither": 1,
            "False": 2
        }
        
        numeric_label = label_mapper[class_label]
        
        elem = list(filter(lambda x: x["feature"] == feature_name, self.pdp))[0]
        grid_values = elem["partial_dependence"]["grid_values"][0]
        average = elem["partial_dependence"]["average"][numeric_label]

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
        