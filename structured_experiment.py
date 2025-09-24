from phoenix.experiments import run_experiment
from experiment_workflow import experiment_workflow
import phoenix as px
import pandas as pd
from prompt_templates.evaluations.truthworthiness import (
    TRUTHWORTHINESS_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.laymans_quality_xai import (
    LAYMAN_TRUTHFULNESS_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.technical_clarity import (
    TECHNICAL_CLARITY_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.technical_clarity_assessment import (
    TECHNICAL_ASSESSMENT_CLARITY_LLM_JUDGE_PROMPT,
)
from prompt_templates.evaluations.focus_quality import (
    FOCUS_QUALITY_LLM_JUDGE_PROMPT,
)
import os
from phoenix.evals import llm_classify, OpenAIModel
from dotenv import load_dotenv
from operations.utils.retrieve_datapoint import retrieve_datapoint

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME_2 = os.getenv("MODEL_NAME_2")
NO_THINKING = os.getenv("NO_THINKING")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")


experiment_version = (
    f"v1.3-{MODEL_NAME}-{'thinking' if NO_THINKING == 'False' else 'no_thinking'}"
)

load_dotenv(override=True)

open_ai_key = os.getenv("API_KEY")
# eval_model = OpenAIModel(model="gpt-4o-mini", api_key=open_ai_key)
eval_model = OpenAIModel(model=MODEL_NAME_2, api_key="ollama", base_url=OLLAMA_ENDPOINT)
full_df = pd.read_csv("data/full_df.csv")


def run_agent_task(dp_id):
    return experiment_workflow(dp_id)


def layman_xai_truthfulness(output: str) -> float:
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
            template=LAYMAN_TRUTHFULNESS_LLM_JUDGE_PROMPT,
            rails=["truthful", "mostly truthful", "untruthful"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "truthful")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


def short_assessment_truthfulness(output: str) -> bool:
    elem = output.get("conclusion")
    df = pd.DataFrame(
        {
            "layman_description": [elem.get("judgement_reason_short")],
            "technical_description": [elem.get("judgement_reason")],
        }
    )
    response = llm_classify(
        data=df,
        template=LAYMAN_TRUTHFULNESS_LLM_JUDGE_PROMPT,
        rails=["truthful", "mostly truthful", "untruthful"],
        model=eval_model,
        provide_explanation=True,
    )

    return response["label"] == "truthful"


def focus_quality(output: str) -> float:
    trace = output.get("trace")
    conclusion = output.get("conclusion")
    relevant_modules = conclusion.get("most_relevant_modules")
    summaries = [
        elem["summary"] for elem in trace if elem.get("module_name") in relevant_modules
    ]
    if len(summaries) == 0:
        return False
    df = pd.DataFrame(
        {
            "natural_language_descriptions": ["\n".join(summaries)],
            "assessment": [conclusion.get("judgement_reason")],
        }
    )
    response = llm_classify(
        data=df,
        template=FOCUS_QUALITY_LLM_JUDGE_PROMPT,
        rails=["strongly contributes", "weakly contributes"],
        model=eval_model,
        provide_explanation=True,
    )

    # output the percentual amount of clear statements
    return response["label"] == "strongly contributes"


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
            rails=["truthful", "mostly truthful", "untruthful"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "truthful")

    # output the percentual amount of truthful statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of xai summaries
def technical_xai_clarity(output: str) -> float:
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
            rails=["clear", "mostly clear", "unclear"],
            model=eval_model,
            provide_explanation=True,
        )
        reasoning.append(response["label"] == "clear")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of technical assessment
def technical_assessment_clarity(output: str) -> float:
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
        rails=["clear", "mostly clear", "unclear"],
        model=eval_model,
        provide_explanation=True,
    )

    return response["label"] == "clear"


def main():
    test_dataset = {
        "pof": [
            {
                "dp_id": 30,
            },
            {
                "dp_id": 31,
            },
            {
                "dp_id": 32,
            },
            {
                "dp_id": 33,
            },
            {
                "dp_id": 34,
            },
            {
                "dp_id": 35,
            },
        ],
        "false": [
            {
                "dp_id": 65,
            },
            {
                "dp_id": 66,
            },
            {
                "dp_id": 67,
            },
            {
                "dp_id": 68,
            },
            {
                "dp_id": 69,
            },
            {"dp_id": 70},
        ],
        "mostly_false": [
            {
                "dp_id": 102,
            },
            {
                "dp_id": 103,
            },
            {
                "dp_id": 104,
            },
            {
                "dp_id": 105,
            },
            {
                "dp_id": 106,
            },
            {
                "dp_id": 107,
            },
        ],
        "half_true": [
            {
                "dp_id": 146,
            },
            {
                "dp_id": 147,
            },
            {
                "dp_id": 148,
            },
            {
                "dp_id": 149,
            },
            {
                "dp_id": 150,
            },
            {
                "dp_id": 151,
            },
        ],
        "mostly_true": [
            {
                "dp_id": 197,
            },
            {
                "dp_id": 198,
            },
            {
                "dp_id": 199,
            },
            {
                "dp_id": 200,
            },
            {
                "dp_id": 201,
            },
            {
                "dp_id": 202,
            },
        ],
        "true": [
            {
                "dp_id": 255,
            },
            {
                "dp_id": 256,
            },
            {
                "dp_id": 257,
            },
            {
                "dp_id": 258,
            },
            {
                "dp_id": 259,
            },
            {
                "dp_id": 260,
            },
        ],
    }
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
    
    for i in range(1, 6):
        for key, value in test_dataset.items():
            ds_version = f"patterns_{key}_v1.1"
            overall_experiment_df = pd.DataFrame(value)
            try:
                dataset = px_client.upload_dataset(dataframe=overall_experiment_df,
                                        dataset_name=f"structured_experiment_inputs_{ds_version}",
                                        input_keys=["dp_id"])
            except Exception as e:
                print(f"Error uploading dataset: {e}")
                dataset = px_client.get_dataset(name=f"structured_experiment_inputs_{ds_version}")
            if not dataset:
                print(f"Error getting dataset: {ds_version}")
                continue
            experiment = run_experiment(
                dataset,
                run_agent_task,
                evaluators=[
                    xai_description_truthfulness,
                    layman_xai_truthfulness,
                    short_assessment_truthfulness,
                    focus_quality,
                    label_correlation,
                    technical_xai_clarity,
                    technical_assessment_clarity,
                ],
                experiment_name=f"Structured Experiment {experiment_version}",
                experiment_description="Evaluating the structured experiment",
            )


if __name__ == "__main__":
    main()
