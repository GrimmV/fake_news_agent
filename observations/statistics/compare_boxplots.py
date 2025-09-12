import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add the paths to import data from both sets
current_dir = os.path.dirname(os.path.abspath(__file__))
set1_path = os.path.join(current_dir, 'set_2')
set2_path = os.path.join(current_dir, 'set_3')

sys.path.append(set1_path)
sys.path.append(set2_path)

# Import all data files from set1
from set_2.focus_quality import focus_quality as set1_focus_quality
from set_2.technical_assessment_clarity import technical_assessment_clarity as set1_technical_assessment_clarity
from set_2.technical_xai_clarity import technical_xai_clarity as set1_technical_xai_clarity
from set_2.xai_description_truthfulness import xai_description_truthfulness as set1_xai_description_truthfulness
from set_2.short_assessment_truthfulness import short_assessment_truthfulness as set1_short_assessment_truthfulness
from set_2.label_correlation import label_correlation as set1_label_correlation
from set_2.layman_xai_truthfulness import layman_xai_truthfulness as set1_layman_xai_truthfulness

# Import all data files from set2
from set_3.focus_quality import focus_quality as set2_focus_quality
from set_3.technical_assessment_clarity import technical_assessment_clarity as set2_technical_assessment_clarity
from set_3.technical_xai_clarity import technical_xai_clarity as set2_technical_xai_clarity
from set_3.xai_description_truthfulness import xai_description_truthfulness as set2_xai_description_truthfulness
from set_3.short_assessment_truthfulness import short_assessment_truthfulness as set2_short_assessment_truthfulness
from set_3.label_correlation import label_correlation as set2_label_correlation
from set_3.layman_xai_truthfulness import layman_xai_truthfulness as set2_layman_xai_truthfulness

def prepare_data_for_comparison(data_dict):
    """
    Convert dictionary data to a format suitable for comparison.
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

def create_comparative_boxplots():
    """
    Create comparative boxplots between set1 and set2 for each measurement property.
    """
    # Define the data files and their display names
    data_files = {
        'Focus Quality': (set1_focus_quality, set2_focus_quality),
        'Technical Assessment Clarity': (set1_technical_assessment_clarity, set2_technical_assessment_clarity),
        'Technical XAI Clarity': (set1_technical_xai_clarity, set2_technical_xai_clarity),
        'XAI Description Truthfulness': (set1_xai_description_truthfulness, set2_xai_description_truthfulness),
        'Short Assessment Truthfulness': (set1_short_assessment_truthfulness, set2_short_assessment_truthfulness),
        'Label Correlation': (set1_label_correlation, set2_label_correlation),
        'Layman XAI Truthfulness': (set1_layman_xai_truthfulness, set2_layman_xai_truthfulness)
    }
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Comparative Boxplots: Set1 vs Set2', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Create comparative boxplots for each measurement property
    for i, (name, (set1_data, set2_data)) in enumerate(data_files.items()):
        if i < len(axes_flat):
            # Prepare data for both sets
            set1_categories, set1_values = prepare_data_for_comparison(set1_data)
            set2_categories, set2_values = prepare_data_for_comparison(set2_data)
            
            # Ensure both sets have the same categories
            all_categories = sorted(set(set1_categories + set2_categories))
            
            # Create data for plotting
            plot_data = []
            plot_labels = []
            plot_colors = []
            
            for category in all_categories:
                # Set1 data
                if category in set1_categories:
                    cat_idx = set1_categories.index(category)
                    plot_data.append(set1_values[cat_idx])
                    plot_labels.append(f'{category}\n(Set1)')
                    plot_colors.append('lightblue')
                
                # Set2 data
                if category in set2_categories:
                    cat_idx = set2_categories.index(category)
                    plot_data.append(set2_values[cat_idx])
                    plot_labels.append(f'{category}\n(Set2)')
                    plot_colors.append('lightcoral')
            
            # Create boxplot
            bp = axes_flat[i].boxplot(plot_data, labels=plot_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize the plot
            axes_flat[i].set_title(name, fontsize=12, fontweight='bold')
            axes_flat[i].set_xlabel('Categories', fontsize=10)
            axes_flat[i].set_ylabel('Values', fontsize=10)
            axes_flat[i].grid(True, alpha=0.3)
            axes_flat[i].tick_params(axis='x', rotation=45, labelsize=8)
            
            # Set y-axis limits to be consistent
            axes_flat[i].set_ylim(0, 1.1)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='Set1'),
                             Patch(facecolor='lightcoral', alpha=0.7, label='Set2')]
            axes_flat[i].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(data_files), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    # plt.savefig('set1_vs_set2_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def create_statistical_comparison():
    """
    Create a detailed statistical comparison between set1 and set2.
    """
    data_files = {
        'Focus Quality': (set1_focus_quality, set2_focus_quality),
        'Technical Assessment Clarity': (set1_technical_assessment_clarity, set2_technical_assessment_clarity),
        'Technical XAI Clarity': (set1_technical_xai_clarity, set2_technical_xai_clarity),
        'XAI Description Truthfulness': (set1_xai_description_truthfulness, set2_xai_description_truthfulness),
        'Short Assessment Truthfulness': (set1_short_assessment_truthfulness, set2_short_assessment_truthfulness),
        'Label Correlation': (set1_label_correlation, set2_label_correlation),
        'Layman XAI Truthfulness': (set1_layman_xai_truthfulness, set2_layman_xai_truthfulness)
    }
    
    print("Statistical Comparison: Set1 vs Set2")
    print("=" * 60)
    
    for name, (set1_data, set2_data) in data_files.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Get all categories
        all_categories = set(set1_data.keys()) | set(set2_data.keys())
        
        for category in sorted(all_categories):
            set1_values = [x for x in set1_data.get(category, []) if x is not None]
            set2_values = [x for x in set2_data.get(category, []) if x is not None]
            
            if len(set1_values) > 0 and len(set2_values) > 0:
                # Calculate statistics
                set1_mean = np.mean(set1_values)
                set2_mean = np.mean(set2_values)
                set1_std = np.std(set1_values)
                set2_std = np.std(set2_values)
                
                # Perform t-test
                try:
                    t_stat, p_value = stats.ttest_ind(set1_values, set2_values)
                    significant = p_value < 0.05
                except:
                    t_stat, p_value = np.nan, np.nan
                    significant = False
                
                # Calculate effect size (Cohen's d)
                try:
                    pooled_std = np.sqrt(((len(set1_values) - 1) * set1_std**2 + 
                                        (len(set2_values) - 1) * set2_std**2) / 
                                       (len(set1_values) + len(set2_values) - 2))
                    cohens_d = (set1_mean - set2_mean) / pooled_std if pooled_std > 0 else 0
                except:
                    cohens_d = 0
                
                print(f"  {category}:")
                print(f"    Set1: mean={set1_mean:.3f}, std={set1_std:.3f}, n={len(set1_values)}")
                print(f"    Set2: mean={set2_mean:.3f}, std={set2_std:.3f}, n={len(set2_values)}")
                print(f"    Difference: {set1_mean - set2_mean:.3f}")
                print(f"    t-test: t={t_stat:.3f}, p={p_value:.3f}, significant={significant}")
                print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
            else:
                print(f"  {category}: Insufficient data for comparison")

def create_side_by_side_heatmap():
    """
    Create a heatmap showing mean values for each category across both sets.
    """
    data_files = {
        'Focus Quality': (set1_focus_quality, set2_focus_quality),
        'Technical Assessment Clarity': (set1_technical_assessment_clarity, set2_technical_assessment_clarity),
        'Technical XAI Clarity': (set1_technical_xai_clarity, set2_technical_xai_clarity),
        'XAI Description Truthfulness': (set1_xai_description_truthfulness, set2_xai_description_truthfulness),
        'Short Assessment Truthfulness': (set1_short_assessment_truthfulness, set2_short_assessment_truthfulness),
        'Label Correlation': (set1_label_correlation, set2_label_correlation),
        'Layman XAI Truthfulness': (set1_layman_xai_truthfulness, set2_layman_xai_truthfulness)
    }
    
    # Get all categories
    all_categories = set()
    for set1_data, set2_data in data_files.values():
        all_categories.update(set1_data.keys())
        all_categories.update(set2_data.keys())
    
    all_categories = sorted(list(all_categories))
    
    # Create data matrix
    heatmap_data = []
    row_labels = []
    
    for name, (set1_data, set2_data) in data_files.items():
        set1_row = []
        set2_row = []
        
        for category in all_categories:
            set1_values = [x for x in set1_data.get(category, []) if x is not None]
            set2_values = [x for x in set2_data.get(category, []) if x is not None]
            
            set1_mean = np.mean(set1_values) if set1_values else np.nan
            set2_mean = np.mean(set2_values) if set2_values else np.nan
            
            set1_row.append(set1_mean)
            set2_row.append(set2_mean)
        
        heatmap_data.append(set1_row)
        heatmap_data.append(set2_row)
        row_labels.append(f'{name} (Set1)')
        row_labels.append(f'{name} (Set2)')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_categories)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    
    # Add values as text
    for i in range(len(row_labels)):
        for j in range(len(all_categories)):
            value = heatmap_data[i][j]
            if not np.isnan(value):
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    ax.set_title('Mean Values Comparison: Set1 vs Set2', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Value', rotation=270, labelpad=15)
    
    plt.tight_layout()
    # plt.savefig('set1_vs_set2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Print statistical comparison
    create_statistical_comparison()
    
    # Create comparative boxplots
    create_comparative_boxplots()
    
    # Create side-by-side heatmap
    create_side_by_side_heatmap()
