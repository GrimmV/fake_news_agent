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

    def get_feature_importance(self, label, **kwargs):

        my_label = label_map[label]

        features = list(self.feature_importances[label].keys())
        values = list(self.feature_importances[label].values())
        
        # Create list of (feature, value) tuples and sort by value
        feature_value_pairs = list(zip(features, values))
        top_features = [feature for feature, _ in sorted(feature_value_pairs, key=lambda x: x[1], reverse=True)[:3]]

        raw = {"label": my_label, "values": self.feature_importances[label], "top_features": top_features}

        visual = px.bar(
            x=features,
            y=values,
            title="Global Feature Importance",
            labels={
                "x": "Features",
                "y": "Importance"
            },
        )

        return {"raw": raw, "visual": visual}

    def get_partial_dependence(self, feature_name, label, **kwargs):

        my_label = label_map[label]

        elem = list(filter(lambda x: x["feature"] == feature_name, self.pdp))[0]
        grid_values = elem["partial_dependence"]["grid_values"][0]
        average = elem["partial_dependence"]["average"][my_label]

        raw = {
            "feature": feature_name,
            "class": label,
            "grid_values": grid_values,
            "average": average,
        }

        visual = px.line(
            x=grid_values,
            y=average,
            title="Partial Dependence Plot",
            labels={
                "x": feature_name,
                "y": "Contribution"
            },
        )

        return {"raw": raw, "visual": visual}


if __name__ == "__main__":

    module = GlobalXAIModule()
    pdp = module.get_partial_dependence("Lexical Diversity (TTR)", "False")

    pdp["visual"].show()
    # print(module.get_feature_importances())
