import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

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

def prepare_data_for_correlation(data_dict):
    """
    Convert dictionary data to a format suitable for correlation analysis.
    Handles None values by replacing them with NaN.
    """
    # Get all categories
    categories = list(data_dict.keys())
    
    # Create a list of lists, handling None values
    data_matrix = []
    for category in categories:
        values = data_dict[category]
        # Replace None with NaN for proper correlation calculation
        processed_values = [np.nan if x is None else x for x in values]
        data_matrix.append(processed_values)
    
    return categories, np.array(data_matrix).T

def calculate_correlation_matrix(data_matrix, method='pearson'):
    """
    Calculate correlation matrix for the data.
    """
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data_matrix)
    
    if method == 'pearson':
        corr_matrix = df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df.corr(method='spearman')
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    return corr_matrix

def create_correlation_heatmap(corr_matrix, title, ax, method='pearson'):
    """
    Create a correlation heatmap for a single dataset.
    """
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values as text
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    # Set title
    ax.set_title(f'{title}\n({method.capitalize()} Correlation)', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

def create_all_correlation_heatmaps():
    """
    Create correlation heatmaps for all data files.
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
    
    # Create a figure with subplots for all heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Correlation Heatmaps for Set1 Statistics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Create heatmaps for each data file
    for i, (name, data) in enumerate(data_files.items()):
        if i < len(axes_flat):
            categories, data_matrix = prepare_data_for_correlation(data)
            
            # Calculate correlation matrix
            corr_matrix = calculate_correlation_matrix(data_matrix, method='pearson')
            
            # Create heatmap
            create_correlation_heatmap(corr_matrix, name, axes_flat[i])
    
    # Hide unused subplots
    for i in range(len(data_files), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    # plt.savefig('set1_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def create_cross_dataset_correlation():
    """
    Create correlation analysis between measurement properties for each label type.
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
    
    # Get all unique label types across all datasets
    all_labels = set()
    for data in data_files.values():
        all_labels.update(data.keys())
    
    all_labels = sorted(list(all_labels))
    
    # Create a large figure for cross-dataset correlations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Correlation Between Measurement Properties for Each Label Type', fontsize=16, fontweight='bold')
    
    # Analyze correlations for each label type
    for idx, label in enumerate(all_labels):
        if idx < 6:  # We have 6 subplots
            ax = axes[idx // 3, idx % 3]
            
            # Collect data for this label from all measurement properties
            label_data = {}
            for prop_name, data in data_files.items():
                if label in data:
                    # Filter out None values
                    values = [x for x in data[label] if x is not None]
                    if len(values) > 1:  # Need at least 2 values for correlation
                        label_data[prop_name] = values
            
            if len(label_data) > 1:
                # Create correlation matrix for this label across measurement properties
                prop_names = list(label_data.keys())
                n_props = len(prop_names)
                corr_matrix = np.ones((n_props, n_props))
                
                for i, prop1 in enumerate(prop_names):
                    for j, prop2 in enumerate(prop_names):
                        if i != j:
                            data1 = label_data[prop1]
                            data2 = label_data[prop2]
                            
                            # Align data lengths (take minimum length)
                            min_len = min(len(data1), len(data2))
                            data1_aligned = data1[:min_len]
                            data2_aligned = data2[:min_len]
                            
                            if len(data1_aligned) > 1:
                                corr, _ = pearsonr(data1_aligned, data2_aligned)
                                corr_matrix[i, j] = corr
                            else:
                                corr_matrix[i, j] = np.nan
                
                # Create heatmap
                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(range(n_props))
                ax.set_yticks(range(n_props))
                ax.set_xticklabels(prop_names, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(prop_names, fontsize=8)
                
                # Add correlation values as text
                for i in range(n_props):
                    for j in range(n_props):
                        value = corr_matrix[i, j]
                        if not np.isnan(value) and i != j:
                            text_color = 'white' if abs(value) > 0.5 else 'black'
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                   color=text_color, fontweight='bold', fontsize=7)
                
                ax.set_title(f'Label: {label.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Correlation', rotation=270, labelpad=15, fontsize=8)
            else:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {label}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Label: {label.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(all_labels), 6):
        axes[idx // 3, idx % 3].set_visible(False)
    
    plt.tight_layout()
    # plt.savefig('set1_property_correlations_by_label.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_correlation_summary():
    """
    Print a summary of correlations between measurement properties for each label.
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
    
    # Get all unique label types
    all_labels = set()
    for data in data_files.values():
        all_labels.update(data.keys())
    
    all_labels = sorted(list(all_labels))
    
    print("Correlation Analysis Summary: Property Correlations by Label Type")
    print("=" * 70)
    
    for label in all_labels:
        print(f"\nLabel: {label.replace('_', ' ').title()}")
        print("-" * 50)
        
        # Collect data for this label from all measurement properties
        label_data = {}
        for prop_name, data in data_files.items():
            if label in data:
                # Filter out None values
                values = [x for x in data[label] if x is not None]
                if len(values) > 1:  # Need at least 2 values for correlation
                    label_data[prop_name] = values
        
        if len(label_data) > 1:
            # Calculate correlations between properties for this label
            prop_names = list(label_data.keys())
            corr_values = []
            pairs = []
            
            for i, prop1 in enumerate(prop_names):
                for j, prop2 in enumerate(prop_names):
                    if i < j:  # Only upper triangle
                        data1 = label_data[prop1]
                        data2 = label_data[prop2]
                        
                        # Align data lengths
                        min_len = min(len(data1), len(data2))
                        data1_aligned = data1[:min_len]
                        data2_aligned = data2[:min_len]
                        
                        if len(data1_aligned) > 1:
                            corr, _ = pearsonr(data1_aligned, data2_aligned)
                            corr_values.append(corr)
                            pairs.append((prop1, prop2))
            
            if corr_values:
                max_corr_idx = np.argmax(np.abs(corr_values))
                print(f"  Strongest correlation: {pairs[max_corr_idx][0]} - {pairs[max_corr_idx][1]}: {corr_values[max_corr_idx]:.3f}")
                print(f"  Mean correlation: {np.mean(corr_values):.3f}")
                print(f"  Std correlation: {np.std(corr_values):.3f}")
                print(f"  Number of property pairs: {len(corr_values)}")
            else:
                print("  No valid correlations found")
        else:
            print("  Insufficient data for correlation analysis")

if __name__ == "__main__":
    # Print correlation summary
    print_correlation_summary()
    
    # Create individual dataset correlation heatmaps
    create_all_correlation_heatmaps()
    
    # Create cross-dataset correlation analysis
    create_cross_dataset_correlation()
