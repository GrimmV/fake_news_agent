import pandas as pd

from descriptions.features import features
from descriptions.labels import labels
from descriptions.module_descriptions import module_descriptions

from llm.llm import GPTModel

from operations.initial import get_relevant_modules


import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv()

API_KEY = os.getenv('API_KEY')
MODEL_NAME = os.getenv("MODEL_NAME")

datapoint_id = 10882

if __name__ == '__main__':
    
    df = pd.read_csv("data/full_df.csv")
    
    row = df[df["id"] == datapoint_id]
    
    prediction_overview = {
        "statement": row["statement"].iloc[0],
        "prediction": int(row["predictions"].iloc[0]),
        "probas": {
            "class_0": float(row['prob_class_0'].iloc[0]),
            "class_1": float(row['prob_class_1'].iloc[0]),
            "class_2": float(row['prob_class_2'].iloc[0]),
        },
        "properties": {
            'Lexical Diversity (TTR)': float(row['Lexical Diversity (TTR)'].iloc[0]),
            'Average Word Length': float(row['Average Word Length'].iloc[0]),
            'Avg Syllables per Word': float(row['Avg Syllables per Word'].iloc[0]),
            'Difficult Word Ratio': float(row['Difficult Word Ratio'].iloc[0]),
            'Dependency Depth': float(row['Dependency Depth'].iloc[0]), 
            'Length': int(row['Length'].iloc[0]), 
            'sentiment': float(row['sentiment'].iloc[0])
        }
    }
    
    llm = GPTModel(model_name=MODEL_NAME, key=API_KEY)
    
    module_list = get_relevant_modules(module_descriptions, prediction_overview, llm, labels, features)
    print(module_list)
    
    