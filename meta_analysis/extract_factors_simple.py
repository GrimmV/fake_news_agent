import json
import pandas as pd


def extract_factors_by_divergence_type(divergence_type, experiment_results_file="experiment_results.json"):
    """
    Extract all factors associated with a specific divergence type.
    
    Args:
        divergence_type (str): The divergence type to filter by (e.g., "explanation_framing")
        experiment_results_file (str): Path to the experiment results JSON file
    
    Returns:
        pd.DataFrame: DataFrame with extracted factors
    """
    
    # Load the experiment results
    with open(experiment_results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    # Collect all factors with the specified divergence type
    matching_factors = []
    
    for experiment_key, cases in all_results.items():
        for case in cases:
            # Only process cases that have analysis
            if case.get('should_investigate') and case.get('analysis') and case.get('case_type'):
                case_type = case['case_type']
                analysis = case['analysis']['analysis']
                factors = analysis.get('factors', [])
                
                # Extract experiment info
                experiment_parts = experiment_key.split('_')
                label = experiment_parts[0]
                exp_id = experiment_parts[2] if len(experiment_parts) > 2 else "unknown"
                
                # Check each factor
                for factor in factors:
                    if factor.get('associated_divergence_type') == divergence_type:
                        matching_factors.append({
                            'experiment_key': experiment_key,
                            'label': label,
                            'experiment_id': exp_id,
                            'case_type': case_type,
                            'explanation': factor.get('explanation', ''),
                            'reference': factor.get('reference', ''),
                            'associated_divergence_type': factor.get('associated_divergence_type', ''),
                            'ground_truth_label': case['analysis'].get('label'),
                            'predicted_label': case['analysis'].get('prediction'),
                            'trustscore': case['analysis'].get('trustscore')
                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(matching_factors)
    
    return df


def get_available_divergence_types(experiment_results_file="experiment_results.json"):
    """
    Get all available divergence types in the experiment results.
    
    Args:
        experiment_results_file (str): Path to the experiment results JSON file
    
    Returns:
        list: List of all unique divergence types
    """
    
    with open(experiment_results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    divergence_types = set()
    
    for experiment_key, cases in all_results.items():
        for case in cases:
            if case.get('should_investigate') and case.get('analysis') and case.get('case_type'):
                factors = case['analysis']['analysis'].get('factors', [])
                for factor in factors:
                    div_type = factor.get('associated_divergence_type')
                    if div_type:
                        divergence_types.add(div_type)
    
    return sorted(divergence_types)


if __name__ == "__main__":
    # Example usage - change the divergence type here
    divergence_type = "explanation_framing"  # Change this to the divergence type you want
    the_id = "improved"
    experiment_results_file = f"experiment_results-{the_id}.json"
    
    
    print(f"Extracting factors for divergence type: '{divergence_type}'")
    
    # Get available divergence types
    available_types = get_available_divergence_types(experiment_results_file)
    print(f"\nAvailable divergence types: {available_types}")
    
    # Extract factors
    df = extract_factors_by_divergence_type(divergence_type, experiment_results_file)
    
    if df.empty:
        print(f"\nNo factors found for divergence type '{divergence_type}'")
    else:
        print(f"\nFound {len(df)} factors for divergence type '{divergence_type}'")
        
        # Save to CSV
        output_file = f"factors_{divergence_type.replace(' ', '_').replace('/', '_')}_{the_id}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to {output_file}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"  Total factors: {len(df)}")
        print(f"  Unique experiments: {df['experiment_key'].nunique()}")
        print(f"  Case types: {df['case_type'].value_counts().to_dict()}")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
        
        # Show first few examples
        print(f"\nFirst 3 factors:")
        for i, row in df.head(3).iterrows():
            print(f"\n{i+1}. Experiment: {row['experiment_key']}")
            print(f"   Explanation: {row['explanation']}")
            print(f"   Reference: {row['reference']}")
            print(f"   Case Type: {row['case_type']}")
