import pandas as pd
import numpy as np
import plotly.express as px

class DistributionModule:
    
    def __init__(self):
        self.df = pd.read_csv("data/full_df.csv")
        
    def get_distribution_1d(self, label, feature_name):
        
        print(label)
        print(feature_name)
        
        class_df = self.df[self.df["new_labels"] == int(label)]
        
        n_bins = 20
        
        counts, bin_edges = np.histogram(class_df[feature_name], bins=n_bins)
        
        raw = {
            "counts": counts,
            "edges": bin_edges
        }
        
        visual = px.histogram(class_df, x=feature_name, nbins=n_bins)
        visual.update_layout(bargap=0.1)
        
        return {
            "raw": raw,
            "visual": visual
        }
        
    def get_distribution_2d(self, label, feature_name_1, feature_name_2):
        
        class_df = self.df[self.df["new_labels"] == label]
        
        col1 = class_df[feature_name_1]
        col2 = class_df[feature_name_2]
        
        counts, xedges, yedges = np.histogram2d(col1, col2, bins=10)
        
        raw = {
            "counts": counts,
            "xedges": xedges,
            "yedges": yedges
        }
        
        visual = px.density_heatmap(class_df, x=feature_name_1, y=feature_name_2)
        
        return {
            "raw": raw,
            "visual": visual
        }
        
        
        
    
if __name__ == "__main__":
    
    dist = DistributionModule()
    distribution = dist.get_distribution_2d(0, "Length", "sentiment")
    
    distribution["visual"].show()
    
    