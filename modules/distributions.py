import pandas as pd
import numpy as np
import plotly.express as px
from modules.utils.label_mapper import label_map
from modules.utils.word_similarity import find_most_similar_word


class DistributionModule:

    def __init__(self, features):
        self.df = pd.read_csv("data/full_df.csv")
        self.features = features

    def get_distribution_1d(self, feature_name, label, **kwargs):

        my_label = label_map[label]

        class_df = self.df[self.df["new_label"] == my_label]
        
        num_unique_values = len(class_df[feature_name].unique())

        n_bins = 20 if num_unique_values > 20 else num_unique_values

        if feature_name not in self.features:
            feature_name = find_most_similar_word(self.features, feature_name)

        counts, bin_edges = np.histogram(class_df[feature_name], bins=n_bins)

        raw = {"counts": counts.astype(int).tolist(), "edges": bin_edges.astype(float).tolist()}

        visual = px.histogram(
            class_df,
            x=feature_name,
            nbins=n_bins,
            title="Feature Distribution",
            # subtitle=f"feature: {feature_name}, label: {label}",
        )
        visual.update_layout(bargap=0.1)

        return {"raw": raw, "visual": visual}

    def get_distribution_2d(self, feature_name_1, feature_name_2, label, **kwargs):

        my_label = label_map[label]

        class_df = self.df[self.df["new_label"] == my_label]

        if feature_name_1 not in self.features:
            feature_name_1 = find_most_similar_word(self.features, feature_name_1)
        if feature_name_2 not in self.features:
            feature_name_2 = find_most_similar_word(self.features, feature_name_2)

        col1 = class_df[feature_name_1]
        col2 = class_df[feature_name_2]

        counts, xedges, yedges = np.histogram2d(col1, col2, bins=10)

        raw = {"counts": counts, "xedges": xedges, "yedges": yedges}

        visual = px.density_heatmap(
            class_df,
            x=feature_name_1,
            y=feature_name_2,
            title="2D Feature Distribution",
            # subtitle=f"features: {feature_name_1}, {feature_name_2}, label: {label}",
        )

        return {"raw": raw, "visual": visual}


if __name__ == "__main__":

    dist = DistributionModule()
    distribution = dist.get_distribution_2d(0, "Length", "sentiment")

    distribution["visual"].show()
