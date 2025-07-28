from phoenix.experiments import run_experiment
from experiment_workflow import experiment_workflow
import phoenix as px
import pandas as pd
from prompt_templates.evaluations.truthworthiness import TRUTHWORTHINESS_LLM_JUDGE_PROMPT
import os
from phoenix.evals import llm_classify, OpenAIModel
from dotenv import load_dotenv

load_dotenv(override=True)

open_ai_key = os.getenv("API_KEY")
eval_model=OpenAIModel(model="gpt-4o-mini", api_key=open_ai_key)

def run_agent_task(dp_id):
    return experiment_workflow(dp_id)
    
# evaluator for tool 2: data analysis
def evaluate_truthworthiness(output: str) -> bool:
    reasoning = []
    for elem in output.get("trace"):
        df = pd.DataFrame({
            "module_output": [elem.get("module_output")],
            "natural_language_description": [elem.get("summary")],
            "function": [elem.get("module_name")],
            "params": [elem.get("module_params")]
        })
        response = llm_classify(
            data=df,
            template=TRUTHWORTHINESS_LLM_JUDGE_PROMPT,
            rails=["truthful", "untruthful"],
            model=eval_model,
            provide_explanation=True
        )
        reasoning.append(response['label'] == 'truthful')
        
    # output the percentual amount of truthful statements
    return sum(reasoning) / len(reasoning)

def main():
    px_client = px.Client()
    # test_dataset = [
    #     {
    #         "dp_id": 1,
    #     },
    #     {
    #         "dp_id": 2, 
    #     },
    #     {
    #         "dp_id": 3,
    #     },
    #     {
    #         "dp_id": 4,
    #     },
    #     {
    #         "dp_id": 5,
    #     },
    # ]
    # overall_experiment_df = pd.DataFrame(test_dataset)
    # dataset = px_client.upload_dataset(dataframe=overall_experiment_df, 
    #                                dataset_name=f"structured_experiment_inputs_v0.1", 
    #                                input_keys=["dp_id"])
    dataset = px_client.get_dataset(name="structured_experiment_inputs_v0.1")
    experiment = run_experiment(
        dataset,
        run_agent_task,
        evaluators=[evaluate_truthworthiness],
        experiment_name="Structured Experiment v0.1",
        experiment_description="Evaluating the structured experiment",
    )

if __name__ == "__main__":
    main()
