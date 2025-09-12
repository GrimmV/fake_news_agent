import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the current directory to the path to import the data files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all data files
from set1.focus_quality import focus_quality
from set1.technical_assessment_clarity import technical_assessment_clarity
from set1.technical_xai_clarity import technical_xai_clarity
from set1.xai_description_truthfulness import xai_description_truthfulness
from set1.short_assessment_truthfulness import short_assessment_truthfulness
from set1.label_correlation import label_correlation
from set1.layman_xai_truthfulness import layman_xai_truthfulness

def prepare_data_for_boxplot(data_dict):
    """
    Convert dictionary data to a format suitable for boxplot.
    Handles None values by filtering them out.
    """
    categories = []
    values = []
    
    for category, data_list in data_dict.items():
        # Filter out None values
        filtered_data = [x for x in data_list if x is not None]
        if filtered_data:  # Only add if there's data
            categories.append(category)
            values.append(filtered_data)
    
    return categories, values

def create_boxplots():
    """
    Create boxplots for all data files in the set1 directory.
    """
    # Define the data files and their display names
    data_files = {
        'Focus Quality': focus_quality,
        'Technical Assessment Clarity': technical_assessment_clarity,
        'Technical XAI Clarity': technical_xai_clarity,
        'XAI Description Truthfulness': xai_description_truthfulness,
        'Short Assessment Truthfulness': short_assessment_truthfulness,
        'Label Correlation': label_correlation,
        'Layman XAI Truthfulness': layman_xai_truthfulness
    }
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Boxplots for Set1 Statistics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Create boxplots for each data file
    for i, (name, data) in enumerate(data_files.items()):
        if i < len(axes_flat):
            categories, values = prepare_data_for_boxplot(data)
            
            # Create boxplot
            bp = axes_flat[i].boxplot(values, labels=categories, patch_artist=True)
            
            # Customize boxplot appearance
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize the plot
            axes_flat[i].set_title(name, fontsize=12, fontweight='bold')
            axes_flat[i].set_xlabel('Categories', fontsize=10)
            axes_flat[i].set_ylabel('Values', fontsize=10)
            axes_flat[i].grid(True, alpha=0.3)
            axes_flat[i].tick_params(axis='x', rotation=45)
            
            # Set y-axis limits to be consistent
            axes_flat[i].set_ylim(0, 1.1)
    
    # Hide the last subplot if we have 7 files but only 6 subplots
    if len(data_files) < len(axes_flat):
        axes_flat[-1].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    # plt.savefig('set1_boxplots.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def print_data_summary():
    """
    Print a summary of the data for each file.
    """
    data_files = {
        'Focus Quality': focus_quality,
        'Technical Assessment Clarity': technical_assessment_clarity,
        'Technical XAI Clarity': technical_xai_clarity,
        'XAI Description Truthfulness': xai_description_truthfulness,
        'Short Assessment Truthfulness': short_assessment_truthfulness,
        'Label Correlation': label_correlation,
        'Layman XAI Truthfulness': layman_xai_truthfulness
    }
    
    print("Data Summary for Set1 Statistics:")
    print("=" * 50)
    
    for name, data in data_files.items():
        print(f"\n{name}:")
        for category, values in data.items():
            # Filter out None values for statistics
            filtered_values = [x for x in values if x is not None]
            if filtered_values:
                print(f"  {category}: {len(filtered_values)} values, "
                      f"mean={np.mean(filtered_values):.3f}, "
                      f"std={np.std(filtered_values):.3f}")
            else:
                print(f"  {category}: No valid data")

if __name__ == "__main__":
    # Print data summary
    print_data_summary()
    
    # Create and display boxplots
    create_boxplots()
