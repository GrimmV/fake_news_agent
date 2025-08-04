from phoenix.experiments import run_experiment
from experiment_workflow import experiment_workflow
import phoenix as px
import pandas as pd
from prompt_templates.evaluations.truthworthiness import (
    TRUTHWORTHINESS_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.laymans_quality_xai import (
    LAYMAN_CLARITY_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.technical_clarity import (
    TECHNICAL_CLARITY_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.technical_clarity_assessment import (
    TECHNICAL_ASSESSMENT_CLARITY_LLM_JUDGE_PROMPT,
)
import os
from phoenix.evals import llm_classify, OpenAIModel
from dotenv import load_dotenv
from operations.utils.retrieve_datapoint import retrieve_datapoint

ds_version = "old_5"
experiment_version = "v1.2"

load_dotenv(override=True)

open_ai_key = os.getenv("API_KEY")
eval_model = OpenAIModel(model="gpt-4o-mini", api_key=open_ai_key)
full_df = pd.read_csv("data/full_df.csv")


def run_agent_task(dp_id):
    return experiment_workflow(dp_id)


def layman_xai_clarity(output: str) -> float:
    reasoning = []
    for elem in output.get("trace"):
        df = pd.DataFrame(
            {
                "layman_description": [elem.get("laymans_summary")],
                "technical_description": [elem.get("summary")],
            }
        )
        response = llm_classify(
            data=df,
            template=LAYMAN_CLARITY_LLM_JUDGE_PROMPT,
            rails=["truthful", "untruthful"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "truthful")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


def short_assessment_clarity(output: str) -> bool:
    elem = output.get("conclusion")
    df = pd.DataFrame(
        {
            "layman_description": [elem.get("judgement_reason_short")],
            "technical_description": [elem.get("judgement_reason")],
        }
    )
    response = llm_classify(
        data=df,
        template=LAYMAN_CLARITY_LLM_JUDGE_PROMPT,
        rails=["truthful", "untruthful"],
        model=eval_model,
        provide_explanation=True,
    )

    return response["label"] == "truthful"


def label_correlation(input: str, output: str) -> float:
    dp_id = input.get("dp_id")
    datapoint = retrieve_datapoint(full_df, dp_id, with_label=True)
    elem = output.get("conclusion")
    judgment_rating = elem.get("judgement_rating")
    true_label = datapoint.get("label")

    # Estimate correlation between judgment rating and true label.
    # If the judgment rating is 0, the true label should be 0.
    # If the judgment rating moves towards 3, the true label should move towards 2.
    return 1 - abs(judgment_rating * 2 - true_label * 3) / 6


# evaluate quality of xai summaries
def xai_description_truthfulness(output: str) -> float:
    reasoning = []
    for elem in output.get("trace"):
        df = pd.DataFrame(
            {
                "module_output": [elem.get("module_output")],
                "natural_language_description": [elem.get("summary")],
                "function": [elem.get("module_name")],
                "params": [elem.get("module_params")],
            }
        )
        response = llm_classify(
            data=df,
            template=TRUTHWORTHINESS_LLM_JUDGE_PROMPT,
            rails=["truthful", "untruthful"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "truthful")

    # output the percentual amount of truthful statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of xai summaries
def technical_clarity(output: str) -> float:
    reasoning = []
    for elem in output.get("trace"):
        df = pd.DataFrame(
            {
                "module_output": [elem.get("module_output")],
                "natural_language_description": [elem.get("summary")],
                "function": [elem.get("module_name")],
                "params": [elem.get("module_params")],
            }
        )
        response = llm_classify(
            data=df,
            template=TECHNICAL_CLARITY_LLM_JUDGE_PROMPT,
            rails=["clear", "unclear"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "clear")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of technical assessment
def technical_assessment_clarity(output: str) -> float:
    reasoning = []
    trace = output.get("trace")
    conclusion = output.get("conclusion")
    summaries = [elem["summary"] for elem in trace]
    df = pd.DataFrame(
        {
            "natural_language_descriptions": ["\n".join(summaries)],
            "assessment": [conclusion.get("judgement_reason")],
        }
    )
    response = llm_classify(
        data=df,
        template=TECHNICAL_ASSESSMENT_CLARITY_LLM_JUDGE_PROMPT,
        rails=["clear", "unclear"],
        model=eval_model,
        provide_explanation=True,
    )
    reasoning.append(response["label"] == "clear")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


def main():
    px_client = px.Client()
    # test_dataset = [
    #     {
    #         "dp_id": 34,
    #     },
    #     {
    #         "dp_id": 68,
    #     },
    #     {
    #         "dp_id": 102,
    #     },
    #     {
    #         "dp_id": 146,
    #     },
    #     {
    #         "dp_id": 197,
    #     },
    #     {
    #         "dp_id": 256,
    #     },
    # ]
    # overall_experiment_df = pd.DataFrame(test_dataset)
    # dataset = px_client.upload_dataset(dataframe=overall_experiment_df,
    #                                dataset_name=f"structured_experiment_inputs_{ds_version}",
    #                                input_keys=["dp_id"])
    dataset = px_client.get_dataset(name=f"structured_experiment_inputs_{ds_version}")
    experiment = run_experiment(
        dataset,
        run_agent_task,
        evaluators=[
            xai_description_truthfulness,
            layman_xai_clarity,
            short_assessment_clarity,
            label_correlation,
            technical_clarity,
            technical_assessment_clarity,
        ],
        experiment_name=f"Structured Experiment {experiment_version}",
        experiment_description="Evaluating the structured experiment",
    )


if __name__ == "__main__":
    main()
