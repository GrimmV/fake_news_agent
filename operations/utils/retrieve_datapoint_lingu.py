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
            "name": "char_count",
            "min": 0,
            "max": None,
            "description": "Total number of characters in the text. Measures the length of the statement.",
        },
        {
            "name": "word_count",
            "min": 0,
            "max": None,
            "description": "Total number of words in the text. Measures the lexical length of the statement.",
        },
        {
            "name": "upper_case_word_count",
            "min": 0,
            "max": None,
            "description": "Number of words written entirely in uppercase letters. Indicates emphasis or shouting.",
        },
        {
            "name": "word_density",
            "min": 0,
            "max": None,
            "description": "Average number of characters per word. Higher values indicate longer, potentially more complex words.",
        },
        {
            "name": "nouns",
            "min": 0,
            "max": None,
            "description": "Count of nouns in the text. Nouns represent entities, objects, or concepts mentioned.",
        },
        {
            "name": "verbs",
            "min": 0,
            "max": None,
            "description": "Count of verbs in the text. Verbs represent actions, states, or processes described.",
        },
        {
            "name": "adjectives",
            "min": 0,
            "max": None,
            "description": "Count of adjectives in the text. Adjectives modify nouns and add descriptive detail.",
        },
        {
            "name": "pronouns",
            "min": 0,
            "max": None,
            "description": "Count of pronouns in the text. Pronouns refer to entities without naming them explicitly.",
        },
        {
            "name": "adverbs",
            "min": 0,
            "max": None,
            "description": "Count of adverbs in the text. Adverbs modify verbs, adjectives, or other adverbs, indicating manner, time, or degree.",
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
