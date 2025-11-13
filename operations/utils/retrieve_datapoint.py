def retrieve_datapoint(df, dp_id, with_label=False):
    row = df[df["id"] == dp_id]

    raw_label = int(row["prediction"].iloc[0])
    true_raw_label = int(row["label"].iloc[0])

    if raw_label == 0:
        label = "False"
    elif raw_label == 1:
        label = "Neither"
    elif raw_label == 2:
        label = "True"
        
    if true_raw_label == 0 or true_raw_label == 1:
        true_label = 0
    elif true_raw_label == 4 or true_raw_label == 5:
        true_label = 2
    else:
        true_label = 1
        
    datapoint_properties = [
        {
            "name": "us_vs_them_lang",
            "min": 0,
            "max": 2,
            "description": "0=Neutral, 1=Moderate rivalry, 2=Extreme demonization. How severely does the post frame opponents as evil/threatening?",
        },
        {
            "name": "exaggerated_uncertainty",
            "min": 0,
            "max": 1,
            "description": "0.0=Speculative, 1.0=Absolute certainty. How definitively are claims presented?",
        },
        {
            "name": "source_quality",
            "min": 0,
            "max": 2,
            "description": "Score 0-2: 2=Specific evidence, 1=Vague sourcing, 0=No evidence. How verifiable are the claims?",
        },
        {
            "name": "victim_villain_language",
            "min": 0,
            "max": 1,
            "description": "0=True, 1=False. Does the post frame an issue as 'good people harmed by evil actors'?",
        },
        {
            "name": "black_and_white_language",
            "min": 0,
            "max": 1,
            "description": "0=True, 1=False. Does the post reduce a complex issue to one cause, two choices, or blame a single group?",
        },
        {
            "name": "dehumanization",
            "min": 0,
            "max": 2,
            "description": "0=Respectful, 1=Negative labeling, 2=Dehumanizing. How are opponents/minorities described?",
        },
        {
            "name": "emotionality",
            "min": -1,
            "max": 1,
            "description": "-1.0=Highly negative, 1.0=Highly positive. How emotionally charged is the text?",
        },
        {
            "name": "reading_difficulty",
            "min": 0,
            "max": 1,
            "description": "0.0=Very easy, 1.0=Very difficult. How accessible is the language used?",
        },
        {
            "name": "sentiment",
            "min": -1,
            "max": 1,
            "description": "-1=Negative, 0=Neutral, 1=Positive. What is the general emotional orientation?",
        },
        {
            "name": "polarization",
            "min": 0,
            "max": 1,
            "description": "0.0=Balanced or nuanced, 1.0=Highly divisive. How strongly does the text separate opposing views?",
        },
    ]

    datapoint = {
        "author": row["speaker"].iloc[0],
        "statement": row["statement"].iloc[0],
        "date": row["date"].iloc[0],
        "avatar": row["avatar"].iloc[0],
        "dp_id": dp_id,
        "prediction": {
            "label": label,
            "probas": {
                "False": "%.2f" % (float(row["prob_class_0"].iloc[0]) * 100) + " %",
                "Neither": "%.2f" % (float(row["prob_class_1"].iloc[0]) * 100) + " %",
                "True": "%.2f" % (float(row["prob_class_2"].iloc[0]) * 100) + " %",
            },
        },
        "properties": {
            elem["name"]: {
                "value": float(row[elem["name"]].iloc[0]),
                "min": elem["min"],
                "max": elem["max"],
                "description": elem["description"],
            } for elem in datapoint_properties
        },
    }
    
    if with_label:
        datapoint["label"] = true_label

    return datapoint
