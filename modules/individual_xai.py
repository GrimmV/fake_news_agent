import json
import plotly.express as px

features = [
    "Lexical Diversity (TTR)",
    "Average Word Length",
    "Avg Syllables per Word",
    "Difficult Word Ratio",
    "Dependency Depth",
    "Length",
    "sentiment",
]


class IndividualXAIModule:

    def __init__(self):
        similars_location = "data/similars.csv"
        cfs_location = "data/counterfactuals.csv"
        shaps_location = "data/shap.csv"

        with open(similars_location) as f:
            self.similars = json.load(f)
        with open(cfs_location) as f:
            self.counterfactuals = json.load(f)
        with open(shaps_location) as f:
            self.shaps = json.load(f)

    def get_similars(self, dp_id):

        elem = list(filter(lambda x: x["id"] == dp_id, self.similars))[0]

        return {"raw": elem, "visual": None}

    def get_counterfactuals(self, dp_id):

        elem = list(filter(lambda x: x["id"] == dp_id, self.counterfactuals))[0]

        return {"raw": elem, "visual": None}

    def get_shap_values(self, dp_id):

        elem = list(filter(lambda x: x["id"] == dp_id, self.shaps))[0]

        value_dict = elem["values"]

        feature_dict = dict(
            (k, value_dict[k]) for k in features if k in value_dict
        )

        elem["values"] = feature_dict

        visual = px.bar(x=feature_dict.keys(), y=feature_dict.values())

        return {
            "raw": elem,
            "visual": visual
        }


if __name__ == "__main__":

    module = IndividualXAIModule()
    print(module.get_similars(10883))
    print(module.get_counterfactuals(10883))
    shap_out = module.get_shap_values(10883)
    shap_out["visual"].show()
