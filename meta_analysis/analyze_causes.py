import json
from collections import defaultdict, Counter
import pandas as pd


def analyze_cause_statistics(experiment_results_file="experiment_results.json"):
    """
    Analyze primary and secondary cause statistics for each case type.
    
    Args:
        experiment_results_file (str): Path to the experiment results JSON file
    
    Returns:
        dict: Statistics for each case type
    """
    
    # Load the experiment results
    with open(experiment_results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    # Initialize data structures for statistics
    case_type_stats = defaultdict(lambda: {
        'total_cases': 0,
        'primary_causes': Counter(),
        'secondary_causes': Counter(),
        'primary_secondary_combinations': Counter(),
        'cases': []
    })
    
    # Process each experiment
    for experiment_key, cases in all_results.items():
        print(f"Processing {experiment_key}: {len(cases)} cases")
        
        for case in cases:
            # Only process cases that have analysis (should_investigate=True)
            if case.get('should_investigate') and case.get('analysis') and case.get('case_type'):
                case_type = case['case_type']
                analysis = case['analysis']['analysis']
                
                # Extract primary and secondary causes
                primary_cause = analysis.get('primary_cause')
                secondary_causes = analysis.get('secondary_causes', [])
                
                if primary_cause:
                    # Update statistics
                    case_type_stats[case_type]['total_cases'] += 1
                    case_type_stats[case_type]['primary_causes'][primary_cause] += 1
                    
                    # Count each secondary cause
                    for secondary_cause in secondary_causes:
                        case_type_stats[case_type]['secondary_causes'][secondary_cause] += 1
                    
                    # Track primary-secondary combinations
                    for secondary_cause in secondary_causes:
                        combination = f"{primary_cause} + {secondary_cause}"
                        case_type_stats[case_type]['primary_secondary_combinations'][combination] += 1
                    
                    # Store the case details
                    case_type_stats[case_type]['cases'].append({
                        'experiment': experiment_key,
                        'primary_cause': primary_cause,
                        'secondary_causes': secondary_causes,
                        'label': case['analysis'].get('label'),
                        'prediction': case['analysis'].get('prediction'),
                        'trustscore': case['analysis'].get('trustscore')
                    })
    
    return dict(case_type_stats)


def generate_summary_report(stats):
    """
    Generate a summary report of the cause statistics.
    
    Args:
        stats (dict): Statistics from analyze_cause_statistics
    
    Returns:
        str: Formatted summary report
    """
    
    report = []
    report.append("=" * 80)
    report.append("CAUSE ANALYSIS SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    for case_type, data in stats.items():
        if data['total_cases'] == 0:
            continue
            
        report.append(f"CASE TYPE: {case_type}")
        report.append("-" * 50)
        report.append(f"Total cases: {data['total_cases']}")
        report.append("")
        
        # Primary causes
        report.append("PRIMARY CAUSES:")
        for cause, count in data['primary_causes'].most_common():
            percentage = (count / data['total_cases']) * 100
            report.append(f"  {cause}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Secondary causes
        report.append("SECONDARY CAUSES:")
        for cause, count in data['secondary_causes'].most_common():
            percentage = (count / data['total_cases']) * 100
            report.append(f"  {cause}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Top primary-secondary combinations
        report.append("TOP PRIMARY-SECONDARY COMBINATIONS:")
        for combo, count in data['primary_secondary_combinations'].most_common(10):
            percentage = (count / data['total_cases']) * 100
            report.append(f"  {combo}: {count} ({percentage:.1f}%)")
        report.append("")
        report.append("=" * 80)
        report.append("")
    
    return "\n".join(report)


def save_detailed_statistics(stats, output_file="cause_statistics.json"):
    """
    Save detailed statistics to a JSON file.
    
    Args:
        stats (dict): Statistics from analyze_cause_statistics
        output_file (str): Output file path
    """
    
    # Convert Counter objects to regular dictionaries for JSON serialization
    json_stats = {}
    for case_type, data in stats.items():
        json_stats[case_type] = {
            'total_cases': data['total_cases'],
            'primary_causes': dict(data['primary_causes']),
            'secondary_causes': dict(data['secondary_causes']),
            'primary_secondary_combinations': dict(data['primary_secondary_combinations']),
            'cases': data['cases']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed statistics saved to {output_file}")


def create_summary_dataframe(stats):
    """
    Create a pandas DataFrame with summary statistics.
    
    Args:
        stats (dict): Statistics from analyze_cause_statistics
    
    Returns:
        pd.DataFrame: Summary statistics DataFrame
    """
    
    summary_data = []
    
    for case_type, data in stats.items():
        if data['total_cases'] == 0:
            continue
            
        # Primary causes summary
        for cause, count in data['primary_causes'].items():
            percentage = (count / data['total_cases']) * 100
            summary_data.append({
                'case_type': case_type,
                'cause_type': 'primary',
                'cause': cause,
                'count': count,
                'percentage': percentage
            })
        
        # Secondary causes summary
        for cause, count in data['secondary_causes'].items():
            percentage = (count / data['total_cases']) * 100
            summary_data.append({
                'case_type': case_type,
                'cause_type': 'secondary',
                'cause': cause,
                'count': count,
                'percentage': percentage
            })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("Analyzing cause statistics from experiment results...")
    
    result_id = "improved"
    
    # Analyze the statistics
    stats = analyze_cause_statistics(f"experiment_results-{result_id}.json")
    
    # Generate and print summary report
    report = generate_summary_report(stats)
    print(report)
    
    # Save detailed statistics
    save_detailed_statistics(stats, f"cause_statistics-{result_id}.json")
    
    # Create and save summary DataFrame
    df = create_summary_dataframe(stats)
    df.to_csv(f"cause_statistics_summary-{result_id}.csv", index=False)
    print("Summary DataFrame saved to cause_statistics_summary.csv")
    
    # Print overall summary
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    total_investigated_cases = sum(data['total_cases'] for data in stats.values())
    print(f"Total investigated cases: {total_investigated_cases}")
    print(f"Number of case types: {len([k for k, v in stats.items() if v['total_cases'] > 0])}")
    
    for case_type, data in stats.items():
        if data['total_cases'] > 0:
            print(f"  {case_type}: {data['total_cases']} cases")
