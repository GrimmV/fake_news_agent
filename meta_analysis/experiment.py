import json
import pandas as pd
from utils.retrieve_datapoint import retrieve_datapoint
from experiment_setup import assess_case

the_id = "improved"
output_file = f"experiment_results-{the_id}.json"

def experiment(experiment_file_path):
    """
    Process an experiment file to extract datapoint information and labels.

    Args:
        experiment_file_path (str): Path to the experiment JSON file (e.g., "false/exp-1.json")

    Returns:
        dict: Dictionary containing dp_id, judgement_rating, predicted_label, and ground_truth_label
    """
    # Load the experiment data
    with open(
        f"observations/experiments_raw/{experiment_file_path}", "r", encoding="utf-8"
    ) as f:
        experiment_data = json.load(f)
        
    assessed_cases = []

    for example in experiment_data:
        # Extract dp_id from the input
        dp_id = example["input"]["dp_id"]

        # Extract judgement_rating from the conclusion
        judgement_rating = example["output"]["conclusion"]["judgement_rating"]

        trace = example["output"]["trace"]
        
        # remove "module_output" and "laymans_summary" from trace (As is not used for the assessment)
        def remove_from_trace(elem):
            elem_copy = elem.copy()
            elem_copy.pop("module_output")
            elem_copy.pop("laymans_summary")
            return elem_copy
        
        trace = [remove_from_trace(elem) for elem in trace]

        # Load the dataframe for retrieve_datapoint
        df = pd.read_csv("data/full_df.csv", encoding="utf-8")

        # Get datapoint information with labels
        datapoint = retrieve_datapoint(df, dp_id, with_label=True)

        # Extract predicted and ground truth labels
        predicted_label = datapoint["prediction"]["label"]
        ground_truth_label = datapoint["label"]

        assessed_case = assess_case(
            trace, ground_truth_label, predicted_label, judgement_rating
        )
        assessed_cases.append(assessed_case)
        
    return assessed_cases


if __name__ == "__main__":
    
    labels = ["pof", "false", "mostly_false", "half_true", "mostly_true", "true"]
    # labels = ["false"]
    # experiment_ids = [1]
    experiment_ids = [21, 22, 23, 24, 25]
    
    # Load existing results or create new dictionary
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded existing results with {len(all_results)} experiments")
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("Starting with empty results dictionary")
    
    for label in labels:
        for experiment_id in experiment_ids:
            experiment_key = f"{label}_exp_{experiment_id}"
            
            # Skip if already processed
            if experiment_key in all_results:
                print(f"Skipping {experiment_key} - already processed")
                continue
            
            try:
                result = experiment(f"{label}/exp-{experiment_id}.json")
                
                # Add result to dictionary
                all_results[experiment_key] = result
                
                # Save results immediately after each experiment
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                
                print(f"Results for {label}/exp-{experiment_id}.json saved to file")
                print(f"Processed {len(result)} cases")
                
            except Exception as e:
                print(f"Error processing {experiment_key}: {e}")
                print("Continuing with next experiment...")
                continue
    
    print(f"\nFinal results saved to {output_file}")
    print(f"Total experiments processed: {len(all_results)}")
