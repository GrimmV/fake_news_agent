import json
import plotly.express as px
import pandas as pd
from descriptions.features import features
from operations.utils.retrieve_datapoint import retrieve_datapoint

features_names = [x["name"] for x in features]

shaps_location = "data/shap.csv"


class IndividualXAIModule:

    def __init__(self):
        similars_location = "data/similars.csv"
        cfs_location = "data/counterfactuals.csv"
        self.df = pd.read_csv("data/full_df.csv")

        with open(similars_location) as f:
            self.similars = json.load(f)
        with open(cfs_location) as f:
            self.counterfactuals = json.load(f)
        # with open(shaps_location) as f:
        #     self.shaps = json.load(f)

    def get_similars(self, dp_id, **kwargs):

        elem = list(filter(lambda x: x["id"] == dp_id, self.similars))[0]
        elem["elems"] = elem["elems"][1:]

        return {"raw": elem, "visual": None}

    def get_counterfactuals(self, dp_id, **kwargs):

        elem = list(filter(lambda x: x["id"] == dp_id, self.counterfactuals))[0]

        return {"raw": elem, "visual": None}

    def get_shap_values(self, dp_id, **kwargs):
        with open(shaps_location) as f:
            shaps = json.load(f)

        # print(shaps)
        print(dp_id)

        elem = list(filter(lambda x: x["id"] == dp_id, shaps))
        datapoint = retrieve_datapoint(self.df, dp_id)
        feature_values = datapoint["properties"]
        print(elem)
        elem = elem[0]

        print("######### normal shap retrieval ##############")
        print(elem)

        value_dict = elem["values"]

        feature_dict = dict(
            (
                k,
                {
                    "shap_value": value_dict[k],
                    "feature_value": feature_values[k]["value"],
                },
            )
            for k in features_names
            if k in value_dict
        )

        elem["values"] = feature_dict
        
        my_list = list(feature_dict.items())
        my_list.sort(key=lambda x: abs(x[1]["shap_value"]), reverse=True)
        top_features = [x[0] for x in my_list]
        elem["top_features"] = top_features

        visual = px.bar(
            x=feature_dict.keys(),
            y=[elem["shap_value"] for elem in feature_dict.values()],
            title="Individual Feature Importance",
            labels={"x": "Features", "y": "Importance"},
        )
        visual.update_xaxes(tickangle=45)
        print(elem)

        return {"raw": elem, "visual": visual}

    def get_word_shap_values(self, dp_id, **kwargs):

        with open(shaps_location) as f:
            shaps = json.load(f)

        elem = list(filter(lambda x: x["id"] == dp_id, shaps))[0]

        print("######### word shap retrieval ##############")
        print(elem)

        value_dict = elem["values"]
        keys = value_dict.keys()
        print(keys)
        print(features_names)
        word_keys = [x for x in keys if x not in features_names]

        feature_dict = dict((k, value_dict[k]) for k in word_keys if k in value_dict)

        elem["values"] = feature_dict


        return {"raw": elem, "visual": None}


if __name__ == "__main__":

    module = IndividualXAIModule()
    print(module.get_similars(10883))
    # print(module.get_counterfactuals(10883))
    shap_out = module.get_shap_values(10883)
    # shap_out = module.get_word_shap_values(10883)
    print(shap_out["raw"])
    # shap_out["visual"].show()
