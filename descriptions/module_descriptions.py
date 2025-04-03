

module_descriptions = [
    {
        "name": "feature distribution",
        "description": "Histogram of the individual feature distribution for a given class.",
        "parameters": [
            "feature_name",
            "label"
        ]
    },
    {
        "name": "feature distribution 2D",
        "description": "Density Heatmap to compare two features for a given class.",
        "parameters": [
            "feature_name_1",
            "feature_name_2",
            "label"
        ]
    },
    # {
    #     "name": "word distribution",
    #     "description": "Histogram of the most used words for a given class.",
    #     "parameters": {
    #         "name": "feature name",
    #         "class": "class label"
    #     }
    # },
    {
        "name": "performance metrics",
        "description": "Metrics to show model performance: accuracy, f1 score, precision, recall and roc auc score",
        "parameters": {}
    },
    {
        "name": "confusion matrix",
        "description": "Number of occurrences for each label-prediction pair.",
        "parameters": {}
    },
    {
        "name": "global feature importance",
        "description": "Global Relevance of the input features.",
        "parameters": [
            "label"
        ]
    },
    {
        "name": "partial dependence plot",
        "description": "How the prediction changes globally based on the feature value.",
        "parameters": [
            "feature_name",
            "label"
        ]
    },
    {
        "name": "individual feature importance",
        "description": "Importance of the features for the individual model prediction at hand.",
        "parameters": {}
    },
    {
        "name": "word importance",
        "description": "Importance of the individual words of the post with respect to the predicted class",
        "parameters": {}
    },
    {
        "name": "similar predictions",
        "description": "Similar datapoints with the same prediction.",
        "parameters": {}
    },
    {
        "name": "counterfactuals",
        "description": "Similar datapoints with another prediction.",
        "parameters": {}
    },
]