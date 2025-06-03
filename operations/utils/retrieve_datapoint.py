def retrieve_datapoint(df, dp_id):
    row = df[df["id"] == dp_id]

    raw_label = int(row["prediction"].iloc[0])

    if raw_label == 0:
        label = "False"
    elif raw_label == 1:
        label = "Neither"
    elif raw_label == 2:
        label = "True"
        
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
            "description": "0=Specific evidence, 1=Vague sourcing, 2=No evidence. How verifiable are the claims?",
        },
        {
            "name": "victim_villain_language",
            "min": 0,
            "max": 1,
            "description": "0=No, 1=Yes. Does the post frame an issue as 'good people harmed by evil actors'?",
        },
        {
            "name": "black_and_white_language",
            "min": 0,
            "max": 1,
            "description": "0=No, 1=Yes. Does the post reduce a complex issue to one cause, two choices, or blame a single group?",
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
            "description": "Emotionality of the text from -1 (very negative) to 1 (very positive)",
        },
        {
            "name": "reading_difficulty",
            "min": 0,
            "max": 1,
            "description": "Reading difficulty of the text from 0 (very easy) to 1 (very difficult)",
        },
        {
            "name": "sentiment",
            "min": -1,
            "max": 1,
            "description": "Overall sentiment of the text (positive (1)/negative (-1)/neutral (0))",
        },
        {
            "name": "polarization",
            "min": 0,
            "max": 1,
            "description": "Polarization of the text from 1 (very polarizing) to 0 (very nuanced)",
        },
    ]

    datapoint = {
        "author": row["speaker"].iloc[0],
        "statement": row["statement"].iloc[0],
        "dp_id": dp_id,
        "prediction": {
            "label": label,
            "probas": {
                "True": "%.2f" % (float(row["prob_class_0"].iloc[0]) * 100) + " %",
                "Neither": "%.2f" % (float(row["prob_class_1"].iloc[0]) * 100) + " %",
                "False": "%.2f" % (float(row["prob_class_2"].iloc[0]) * 100) + " %",
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

    return datapoint
