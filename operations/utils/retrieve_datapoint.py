def retrieve_datapoint(df, dp_id):
    row = df[df["id"] == dp_id]

    raw_label = int(row["predictions"].iloc[0])

    if raw_label == 0:
        label = "True"
    elif raw_label == 1:
        label = "Neither"
    elif raw_label == 2:
        label = "False"

    lex_diversity_min = df["Lexical Diversity (TTR)"].min()
    lex_diversity_max = df["Lexical Diversity (TTR)"].max()
    word_length_min = df["Average Word Length"].min()
    word_length_max = df["Average Word Length"].max()
    syllables_min = df["Avg Syllables per Word"].min()
    syllables_max = df["Avg Syllables per Word"].max()
    difficult_word_min = df["Difficult Word Ratio"].min()
    difficult_word_max = df["Difficult Word Ratio"].max()
    dependency_min = df["Dependency Depth"].min()
    dependency_max = df["Dependency Depth"].max()
    length_min = df["Length"].min()
    length_max = df["Length"].max()
    sentiment_min = df["sentiment"].min()
    sentiment_max = df["sentiment"].max()

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
            "Lexical Diversity (TTR)": {
                "value": float(row["Lexical Diversity (TTR)"].iloc[0]),
                "min": lex_diversity_min,
                "max": lex_diversity_max,
                "description": "Measures the range of vocabulary used, calculated as the ratio of unique words to total words (Type-Token Ratio).",
            },
            "Average Word Length": {
                "value": float(row["Average Word Length"].iloc[0]),
                "min": word_length_min,
                "max": word_length_max,
                "description": "Indicates the average number of characters per word, reflecting the complexity of word choice.",
            },
            "Avg Syllables per Word": {
                "value": float(row["Avg Syllables per Word"].iloc[0]),
                "min": syllables_min,
                "max": syllables_max,
                "description": "Represents the average number of syllables per word, often used to estimate reading difficulty.",
            },
            "Difficult Word Ratio": {
                "value": float(row["Difficult Word Ratio"].iloc[0]),
                "min": difficult_word_min,
                "max": difficult_word_max,
                "description": "The proportion of words considered difficult based on standard readability word lists.",
            },
            "Dependency Depth": {
                "value": float(row["Dependency Depth"].iloc[0]),
                "min": dependency_min,
                "max": dependency_max,
                "description": "Captures the syntactic complexity of a sentence by measuring the average depth of its dependency parse tree.",
            },
            "Length": {
                "value": int(row["Length"].iloc[0]),
                "min": length_min,
                "max": length_max,
                "description": "Total number of words or tokens in the text, indicating its overall size.",
            },
            "sentiment": {
                "value": float("%.2f" % (float(row["sentiment"].iloc[0]))),
                "min": sentiment_min,
                "max": 1,
                "description": "Sentiment polarity score ranging from negative to positive, with higher values indicating more positive sentiment.",
            },
        },
    }

    return datapoint
