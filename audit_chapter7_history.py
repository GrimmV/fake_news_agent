"""Audit archived Chapter 7 data without calling a model or changing source outputs.

Run with .venv/Scripts/python.exe audit_chapter7_history.py.
Writes a JSON audit and a per-observation CSV under figures/chapter7.
"""

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures/chapter7"
LABELS = ["pof", "false", "mostly_false", "half_true", "mostly_true", "true"]
MODEL = "gpt-5-mini-2025-08-07"
A = "model_wrong_llm_trusts"
B = "model_right_llm_distrusts"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def literal(path):
    return ast.literal_eval(ast.parse(path.read_text(encoding="utf-8-sig")).body[0].value)


def trace_key(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def kind(y, p, t):
    gap = abs(y - p)
    return A if gap >= 1 and t >= 1 else B if gap <= 0 and t <= 0 else None


def load_function(path, name):
    """Load only a named function; do not execute the source module's imports."""
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    with (ROOT / "data/tone_rf_train-gpt-5-mini-eval-1.csv").open(encoding="utf-8-sig") as f:
        dataset = {int(row["id"]): row for row in csv.DictReader(f)}
    formula = load_function(ROOT / "structured_experiment.py", "label_correlation")
    # Reproduce the label lookup with the archived CSV, without importing the application.
    formula.__globals__["full_df"] = dataset
    formula.__globals__["retrieve_datapoint"] = lambda df, dp_id, with_label: {
        "label": int(df[dp_id]["label"]) // 2
    }
    prompt = load_function(ROOT / "meta_analysis/prompts/divergence_analysis.py", "divergence_analysis_prompt")
    observations, trace_index = [], defaultdict(list)
    audit = {"method": "Read-only reconstruction from raw exports, saved results, CSV labels/predictions, and logged requests; no model calls.",
             "conditions": {}}
    saved_results = {}
    for condition, config, runs in [("initial", "30b-instruct", range(1, 6)),
                                     ("improved", "30b-improved", range(36, 41))]:
        saved = read_json(ROOT / f"experiment_results-{condition}_v1.5.json")
        saved_results[condition] = saved
        combined = literal(ROOT / f"observations/statistics_v1-5/combined/set-{config}/label_correlation.py")
        formula_errors, array_errors, branch_errors = [], [], []
        all_scores, raw_scores, label_means = [], [], {}
        for label in LABELS:
            label_values = []
            for run_index, run in enumerate(runs):
                source = f"observations/experiments_raw_v1.5/{label}/exp-{run}.json"
                source_text = (ROOT / source).read_text(encoding="utf-8")
                rows = json.loads(source_text)
                input_lines = [n for n, line in enumerate(source_text.splitlines(), 1) if '"dp_id":' in line]
                assert len(rows) == len(input_lines) == 17
                group = f"{label}_exp_{run}"
                per_run = literal(ROOT / f"observations/statistics_v1-5/set_{run}/label_correlation.py")[label]
                if group in saved:
                    assert len(saved[group]) == 17
                for index, raw in enumerate(rows):
                    dp_id = raw["input"]["dp_id"]
                    y_raw = int(dataset[dp_id]["label"])
                    assert y_raw == LABELS.index(label)
                    y, p = y_raw // 2, int(dataset[dp_id]["prediction"])
                    t = raw["output"]["conclusion"]["judgement_rating"]
                    score = formula(raw["input"], raw["output"])
                    annotation = [a for a in raw["annotations"] if a["name"] == "label_correlation"]
                    assert len(annotation) == 1
                    if score != annotation[0]["score"]:
                        formula_errors.append([group, index])
                    rounded = float(f"{score:.2f}")
                    if rounded != per_run[index] or rounded != combined[label][run_index][index]:
                        array_errors.append([group, index])
                    label_values.append(rounded)
                    raw_scores.append(score)
                    row = dict(condition=condition, group=group, index=index, dp_id=dp_id,
                               raw_source=source, raw_input_line=input_lines[index],
                               label_raw=y_raw, label_mapped=y, prediction=p, trustscore=t,
                               formula_score=score, figure_score=rounded,
                               diagnostic_batch_saved=group in saved, reconstructed_case_type=kind(y, p, t),
                               request_log_lines=[], request_message_counts=[])
                    if group in saved:
                        result = saved[group][index]
                        if result["case_type"] != row["reconstructed_case_type"] or result["should_investigate"] != bool(row["reconstructed_case_type"]):
                            branch_errors.append([group, index, "branch"])
                        if result["analysis"]:
                            for field, value in [("label", y), ("prediction", p), ("trustscore", t)]:
                                if result["analysis"][field] != value:
                                    branch_errors.append([group, index, field])
                        else:
                            assert result["rationale"] == f"No trigger: gap={abs(y-p):.2f} (thr align≤0, diverge≥1), trust={t} (low≤0, high≥1)."
                    trace = []
                    for item in raw["output"]["trace"]:
                        item = item.copy()
                        item.pop("module_output")
                        item.pop("laymans_summary")
                        trace.append(item)
                    row["trace_sha256"] = trace_key(trace)
                    trace_index[row["trace_sha256"]].append(row)
                    observations.append(row)
            label_means[label] = float(np.mean(label_values))
            all_scores.extend(label_values)
        # Match the saved combined file's label order for the exact floating point result.
        mean = float(np.mean([label_means[label] for label in combined]))
        cases = [c for batch in saved.values() for c in batch]
        flags = [c for c in cases if c["should_investigate"]]
        cause_summary = read_json(ROOT / f"cause_statistics-{condition}_v1.5.json")
        assert Counter(c["case_type"] for c in flags) == Counter({k: v["total_cases"] for k, v in cause_summary.items()})
        audit["conditions"][condition] = dict(
            raw_observations=len(all_scores), raw_trust_counts=dict(Counter(o["trustscore"] for o in observations if o["condition"] == condition)),
            formula_mismatches=formula_errors, rounded_array_mismatches=array_errors,
            branch_or_input_mismatches=branch_errors, label_means=label_means,
            figure_mean=mean, unrounded_formula_mean=float(np.mean(raw_scores)),
            saved_batches=len(saved), evaluated_assessments=len(cases),
            flagged=len(flags), directions=dict(Counter(c["case_type"] for c in flags)),
            unflagged=len(cases)-len(flags),
            flagged_missing_analysis=sum(not c.get("analysis") for c in flags),
            flagged_missing_primary=sum(not c.get("analysis", {}).get("analysis", {}).get("primary_cause") for c in flags))
        assert not formula_errors and not array_errors and not branch_errors

    requests, other_requests = [], Counter()
    with (ROOT / "app.log").open(encoding="cp1252") as f:
        for line_number, line in enumerate(f, 1):
            if "Function called with kwargs:" not in line or "expert analyst of LLM behavior" not in line:
                continue
            kwargs = ast.literal_eval(line.split("Function called with kwargs: ", 1)[1])
            content = kwargs["messages"][1]["content"]
            trace_text = content.split("<TRACE>")[1].split("</TRACE>")[0].strip()
            if not trace_text:
                other_requests["empty_trace"] += 1
                continue
            trace = ast.literal_eval(trace_text)
            t = int(re.search(r"Trustscore: (\d+)", content)[1])
            y = int(re.search(r"Label: (\d+)", content)[1])
            p = int(re.search(r"Prediction: (\d+)", content)[1])
            candidates = [o for o in trace_index[trace_key(trace)] if o["trustscore"] == t]
            if kwargs["model"] != MODEL:
                other_requests[kwargs["model"]] += 1
                continue
            assert len(candidates) == 1, (line_number, len(candidates))
            row = candidates[0]
            assert (y, p) == (row["label_mapped"], row["prediction"])
            case_type = kind(y, p, t)
            assert case_type is not None
            assert content == prompt(trace, case_type, y, p, t), line_number
            row["request_log_lines"].append(line_number)
            row["request_message_counts"].append(len(kwargs["messages"]))
            corrections = [m["content"] for m in kwargs["messages"][2:] if m["role"] == "tool"]
            requests.append(dict(line=line_number, timestamp=line[:23], condition=row["condition"],
                                 group=row["group"], index=row["index"], dp_id=row["dp_id"],
                                 model=kwargs["model"], case_type=case_type, trustscore=t,
                                 message_count=len(kwargs["messages"]), prompt_matches_source=True,
                                 scale_description="0 (poor) to 3 (excellent)", assume_prediction_correct=True,
                                 validation_corrections=corrections))

    audit["other_logged_divergence_requests"] = dict(other_requests)
    audit["logged_requests"] = requests
    for condition, info in audit["conditions"].items():
        obs = [o for o in observations if o["condition"] == condition]
        saved = saved_results[condition]
        flag_obs = [o for o in obs if o["diagnostic_batch_saved"] and o["reconstructed_case_type"]]
        assert all(o["request_log_lines"] for o in flag_obs)
        req = [r for r in requests if r["condition"] == condition]
        info["judge_requests"] = len(req)
        info["judge_unique_observations"] = sum(bool(o["request_log_lines"]) for o in obs)
        info["judge_unique_trust_counts"] = dict(Counter(o["trustscore"] for o in obs if o["request_log_lines"]))
        info["judge_request_trust_counts_including_retries"] = dict(Counter(r["trustscore"] for r in req))
        info["saved_flags_with_request_match"] = len(flag_obs)
        info["saved_flags_with_retries"] = sum(len(o["request_log_lines"]) > 1 for o in flag_obs)
        info["saved_flags_by_number_of_attempts"] = dict(Counter(len(o["request_log_lines"]) for o in flag_obs))
        info["full_510_reconstructed_directions"] = dict(Counter(o["reconstructed_case_type"] or "not_flagged" for o in obs))
        missing = []
        missing_stages = {s: Counter() for s in ["before_terminal_case", "terminal_case", "after_terminal_case"]}
        for group in dict.fromkeys(o["group"] for o in obs if not o["diagnostic_batch_saved"]):
            rows = [o for o in obs if o["group"] == group]
            attempted = [o for o in rows if o["request_log_lines"]]
            last = max(attempted, key=lambda o: o["request_log_lines"][-1])
            last_request = next(r for r in req if r["line"] == last["request_log_lines"][-1])
            for row in rows:
                stage = ("before_terminal_case" if row["index"] < last["index"] else
                         "terminal_case" if row["index"] == last["index"] else "after_terminal_case")
                missing_stages[stage][row["reconstructed_case_type"] or "not_flagged"] += 1
            # A request log preserves earlier invalid responses in retry messages;
            # it does not preserve the terminal response or the printed exception.
            missing.append(dict(group=group, missing_observations=len(rows),
                                reconstructed_directions=dict(Counter(o["reconstructed_case_type"] or "not_flagged" for o in rows)),
                                judge_attempted_observations=len(attempted),
                                preceding_rows_in_batch=last["index"],
                                terminal_case_index=last["index"], terminal_dp_id=last["dp_id"],
                                cases_after_terminal_request=16-last["index"],
                                terminal_request_lines=last["request_log_lines"],
                                terminal_message_counts=last["request_message_counts"],
                                recorded_validation_corrections=last_request["validation_corrections"],
                                conclusion="Entire 17-row batch absent. Requests stop at this case after three attempts with preceding schema validation failures. Terminal failure is inferred from control flow; final response/printed exception is not archived."))
        info["missing_batches"] = missing
        info["missing_batch_stage_reconstruction"] = {
            stage: {"rows": sum(counts.values()), "directions": dict(counts)}
            for stage, counts in missing_stages.items()
        }
    audit["limitations"] = [
        "Logged retry messages preserve validation failures on earlier attempts, but terminal responses and the caller's printed exception messages are not in app.log.",
        "Prior-row completion and later-row nonexecution in absent batches are reconstructed from sequential caller control flow and request order, not a persisted row-status manifest.",
        "All 139 saved flags have exact trace/input/prompt request matches for the gpt-5-mini runs; saved result files lack request IDs or commit/prompt hashes.",
        "Full-510 branch counts are a deterministic reconstruction, not additional historical judge diagnoses."
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "historical_trace_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "historical_trace_observations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    for condition, info in audit["conditions"].items():
        print(condition, json.dumps({k: v for k, v in info.items() if k not in {"missing_batches", "label_means"}}))
        for batch in info["missing_batches"]:
            print(json.dumps({k: v for k, v in batch.items() if k not in {"recorded_validation_corrections", "conclusion"}}))


if __name__ == "__main__":
    main()
