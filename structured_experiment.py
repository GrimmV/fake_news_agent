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
from experimentation.response_models.xai_description_truthfulness import XaiDescriptionTruthfulness
from experimentation.response_models.layman_xai_truthfulness import LaymanXaiTruthfulness
from experimentation.response_models.short_assessment_truthfulness import ShortAssessmentTruthfulness
from experimentation.response_models.focus_quality import FocusQuality
from experimentation.response_models.technical_xai_clarity import TechnicalXaiClarity
from experimentation.response_models.technical_assessment_clarity import TechnicalAssessmentClarity
import os
from dotenv import load_dotenv
from operations.utils.retrieve_datapoint import retrieve_datapoint
from llm.llm import GPTModel, OllamaOpenAI

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME_2 = os.getenv("MODEL_NAME_2")
NO_THINKING = os.getenv("NO_THINKING")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT_2")


experiment_version = (
    f"v3.0-{MODEL_NAME}-{'thinking' if NO_THINKING == 'False' else 'no_thinking'}"
)

load_dotenv(override=True)

open_ai_key = os.getenv("API_KEY")
# eval_model = OpenAIModel(model="gpt-4o-mini", api_key=open_ai_key)
# eval_model = OpenAIModel(model=MODEL_NAME_2, api_key="ollama", base_url=OLLAMA_ENDPOINT)
if any([elem in MODEL_NAME for elem in ["gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "gpt-5-mini", "gpt-4.1"]]):
    eval_model = GPTModel(model_name=MODEL_NAME_2, key=open_ai_key)
else:
    eval_model = OllamaOpenAI(model_name=MODEL_NAME_2, base_url=OLLAMA_ENDPOINT)
full_df = pd.read_csv("data/tone_rf_train-gpt-5-mini-eval-1.csv")


def run_agent_task(dp_id):
    # return [{ "content": { "answer": f"test and response" } }]
    return experiment_workflow(dp_id)


def layman_xai_truthfulness(output: str) -> float:
    reasoning = []
    for elem in output.get("trace"):
        response = eval_model.generate(
            prompt=LAYMAN_TRUTHFULNESS_LLM_JUDGE_PROMPT.format(technical_description=elem.get("summary"), layman_description=elem.get("laymans_summary")),
            response_model=LaymanXaiTruthfulness
        )
        reasoning.append(response.truthfulness == "truthful")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


def short_assessment_truthfulness(output: str) -> bool:
    elem = output.get("conclusion")
    response = eval_model.generate(
        prompt=LAYMAN_TRUTHFULNESS_LLM_JUDGE_PROMPT.format(technical_description=elem.get("judgement_reason"), layman_description=elem.get("judgement_reason_short")),
        response_model=ShortAssessmentTruthfulness
    )

    return response.truthfulness == "truthful"


def focus_quality(output: str) -> float:
    trace = output.get("trace")
    conclusion = output.get("conclusion")
    relevant_modules = conclusion.get("most_relevant_modules")
    summaries = [
        elem["summary"] for elem in trace if elem.get("module_name") in relevant_modules
    ]
    if len(summaries) == 0:
        return False
    response = eval_model.generate(
        prompt=FOCUS_QUALITY_LLM_JUDGE_PROMPT.format(natural_language_descriptions="\n".join(summaries), assessment=conclusion.get("judgement_reason")),
        response_model=FocusQuality
    )

    # output the percentual amount of clear statements
    return response.focus_quality == "strongly contributes"


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
        response = eval_model.generate(
            prompt=TRUTHWORTHINESS_LLM_JUDGE_PROMPT.format(module_output=elem.get("module_output"), natural_language_description=elem.get("summary"), function=elem.get("module_name"), params=elem.get("module_params")),
            response_model=XaiDescriptionTruthfulness
        )
        reasoning.append(response.truthfulness == "truthful")

    # output the percentual amount of truthful statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of xai summaries
def technical_xai_clarity(output: str) -> float:
    reasoning = []
    for elem in output.get("trace"):
        response = eval_model.generate(
            prompt=TECHNICAL_CLARITY_LLM_JUDGE_PROMPT.format(module_output=elem.get("module_output"), natural_language_description=elem.get("summary"), function=elem.get("module_name"), params=elem.get("module_params")),
            response_model=TechnicalXaiClarity
        )
        reasoning.append(response.clarity == "clear")

    # output the percentual amount of clear statements
    return sum(reasoning) / len(reasoning)


# evaluate quality of technical assessment
def technical_assessment_clarity(output: str) -> float:
    trace = output.get("trace")
    conclusion = output.get("conclusion")
    summaries = [elem["summary"] for elem in trace]
    response = eval_model.generate(
        prompt=TECHNICAL_ASSESSMENT_CLARITY_LLM_JUDGE_PROMPT.format(natural_language_descriptions="\n".join(summaries), assessment=conclusion.get("judgement_reason")),
        response_model=TechnicalAssessmentClarity
    )

    return response.clarity == "clear"


def main():
    pof = [{"dp_id": elem} for elem in [0, 4, 5, 7, 9, 10, 11, 12, 15, 16, 18, 22, 26, 28, 30, 31, 33]]
    false = [{"dp_id": elem} for elem in [46, 50, 51, 53, 55, 56, 57, 58, 61, 62, 64, 68, 72, 74, 76, 77, 79]]
    mostly_false = [{"dp_id": elem} for elem in [92, 96, 97, 99, 101, 102, 103, 104, 107, 108, 110, 114, 118, 120, 122, 123, 125]]
    half_true = [{"dp_id": elem} for elem in [138, 142, 143, 145, 147, 148, 149, 150, 153, 154, 156, 160, 164, 166, 168, 169, 171]]
    mostly_true = [{"dp_id": elem} for elem in [184, 188, 189, 191, 193, 194, 195, 196, 199, 200, 202, 206, 210, 212, 214, 215, 217]]
    true = [{"dp_id": elem} for elem in [230, 234, 235, 237, 239, 240, 241, 242, 245, 246, 248, 252, 256, 258, 260, 261, 263]]
    test_dataset = {
        "pof": pof,
        "false": false,
        "mostly_false": mostly_false,
        "half_true": half_true,
        "mostly_true": mostly_true,
        "true": true
    }
    px_client = px.Client()
    
    for i in range(1, 6):
        for key, value in test_dataset.items():
            ds_version = f"patterns_{key}_v1.5"
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
                concurrency=5,
                timeout=100000000,
            )


if __name__ == "__main__":
    main()
