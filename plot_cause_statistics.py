import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Configuration
FILES = [
    "cause_statistics_summary-initial.csv",
    "cause_statistics_summary-improved.csv"
]

OUTPUT_PREFIX = "cause_analysis"
FIGSIZE = (12, 8)
BAR_WIDTH = 0.35
FONTSIZE_TITLE = 16
FONTSIZE_LABELS = 14
FONTSIZE_TICKS = 14

# Colors for primary and secondary causes
SECONDARY_COLOR = "#f0d9a8"
PRIMARY_COLOR = "#bcd7f5"

cause_mapping = {
    "explanation_framing": "Explanation Framing",
    "performance_baseline": "Performance Baseline",
    "overgeneralization_from_dataset_statistics": "Overgeneralization",
    "meta_performance_overweighting": "Performance Overweighting",
    "content_model_confusion": "Content Confusion",
    "feature_interpretation_bias": "Interpretation Bias",
    "label_trust_mismatch": "Label Trust Mismatch",
}

def load_and_process_data(filepath: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load CSV data and separate primary and secondary causes with their counts, merging across case types."""
    df = pd.read_csv(filepath)
    
    # Group by cause_type and cause, then sum the counts across all case_types
    primary_causes = df[df['cause_type'] == 'primary'].groupby('cause')['count'].sum().to_dict()
    secondary_causes = df[df['cause_type'] == 'secondary'].groupby('cause')['count'].sum().to_dict()
    
    return primary_causes, secondary_causes

def create_single_subplot(ax, primary_data: Dict[str, int], secondary_data: Dict[str, int], 
                         title: str, all_causes: List[str], show_ylabel: bool = False) -> None:
    """Create a single subplot for one dataset."""
    
    # Prepare data for plotting
    primary_counts = [primary_data.get(cause, 0) for cause in all_causes]
    secondary_counts = [secondary_data.get(cause, 0) for cause in all_causes]
    
    # Set up x positions for bars
    x = np.arange(len(all_causes))
    
    # Create bars
    bars1 = ax.bar(x - BAR_WIDTH/2, primary_counts, BAR_WIDTH, 
                   label='Primary Causes', color=PRIMARY_COLOR, alpha=0.8)
    bars2 = ax.bar(x + BAR_WIDTH/2, secondary_counts, BAR_WIDTH, 
                   label='Secondary Causes', color=SECONDARY_COLOR, alpha=0.8)
    
    # Customize the plot
    # ax.set_xlabel('Causes', fontsize=FONTSIZE_LABELS, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=FONTSIZE_LABELS, fontweight='bold')
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([cause_mapping[cause] for cause in all_causes], rotation=45, fontsize=FONTSIZE_TICKS)
    
    # Format y-axis to remove decimal places
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)
    
    # Add legend (moved slightly to the left)
    ax.legend(fontsize=FONTSIZE_LABELS, loc='upper left')
    
    # Set y-axis limit
    ax.set_ylim(0, 33)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)

def main():
    """Main function to generate a single plot with two side-by-side subplots."""
    
    # Load data from both files
    datasets = []
    all_causes_set = set()
    
    for filename in FILES:
        print(f"Processing {filename}...")
        primary_data, secondary_data = load_and_process_data(filename)
        dataset_name = filename.replace('cause_statistics_summary-', '').replace('.csv', '')
        
        datasets.append({
            'name': dataset_name,
            'primary': primary_data,
            'secondary': secondary_data
        })
        
        # Collect all causes
        all_causes_set.update(primary_data.keys())
        all_causes_set.update(secondary_data.keys())
        
        # Print summary statistics
        print(f"Primary causes: {len(primary_data)} unique causes")
        print(f"Secondary causes: {len(secondary_data)} unique causes")
        print(f"Total primary count: {sum(primary_data.values())}")
        print(f"Total secondary count: {sum(secondary_data.values())}")
    
    # Sort all causes by primary count in descending order (using first dataset as reference)
    all_causes = sorted(all_causes_set, key=lambda x: datasets[0]['primary'].get(x, 0), reverse=True)
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create subplots
    for i, dataset in enumerate(datasets):
        title = f"{dataset['name'].title()}"
        show_ylabel = (i == 0)  # Only show y-label on the first subplot
        create_single_subplot(axes[i], dataset['primary'], dataset['secondary'], title, all_causes, show_ylabel)
    
    # Add main title
    fig.suptitle("Cause Analysis Comparison", fontsize=20, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the combined plot
    output_filename = f"{OUTPUT_PREFIX}_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot: {output_filename}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
