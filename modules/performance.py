import json
import plotly.express as px

class PerformanceModule:
    
    def __init__(self):
        metrics_location = "data/metrics.csv"
        confusion_location = "data/confusion.csv"
        
        with open(metrics_location) as f:
            self.metrics = json.load(f)
        with open(confusion_location) as f:
            self.confusion = json.load(f)
        
        
    def get_performances(self):
        
        visual = px.bar(x=self.metrics.keys(), y=self.metrics.values())
        
        return {
            "raw": self.metrics,
            "visual": visual
        }
    
    def get_confusion(self):
        
        visual = px.imshow(self.confusion, text_auto=True)
        
        return {
            "raw": self.confusion,
            "visual": visual
        }
    
if __name__ == "__main__":
    
    module = PerformanceModule()
    metrics = module.get_performances()
    # metrics["visual"].show()
    conf = module.get_confusion()
    conf["visual"].show()