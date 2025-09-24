import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_cause_statistics(csv_file_path: str, title: str = "Cause Statistics") -> None:
    """
    Generate a bar chart showing primary and secondary amounts for each cause,
    ordered by primary count.
    
    Args:
        csv_file_path (str): Path to the CSV file
        title (str): Title for the plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Group by cause and cause_type, then sum the counts
    aggregated = df.groupby(['cause', 'cause_type'])['count'].sum().reset_index()
    
    # Separate primary and secondary causes
    primary_df = aggregated[aggregated['cause_type'] == 'primary'].copy()
    secondary_df = aggregated[aggregated['cause_type'] == 'secondary'].copy()
    
    # Create a dictionary to map causes to their counts
    primary_counts = dict(zip(primary_df['cause'], primary_df['count']))
    secondary_counts = dict(zip(secondary_df['cause'], secondary_df['count']))
    
    # Get all unique causes and sort by primary count (descending)
    all_causes = set(primary_counts.keys()) | set(secondary_counts.keys())
    sorted_causes = sorted(all_causes, key=lambda x: primary_counts.get(x, 0), reverse=True)
    
    # Prepare data for plotting
    primary_values = [primary_counts.get(cause, 0) for cause in sorted_causes]
    secondary_values = [secondary_counts.get(cause, 0) for cause in sorted_causes]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(sorted_causes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, primary_values, width, label='Primary', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, secondary_values, width, label='Secondary', alpha=0.8, color='#ff7f0e')
    
    # Customize the plot
    ax.set_xlabel('Causes')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_causes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if height is greater than 0
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.show()

def plot_compare_cause_statistics(csv_file1: str, csv_file2: str, 
                                title1: str = "Dataset 1", title2: str = "Dataset 2") -> None:
    """
    Create side-by-side subplots comparing cause statistics from two CSV files.
    
    Args:
        csv_file1 (str): Path to the first CSV file
        csv_file2 (str): Path to the second CSV file
        title1 (str): Title for the first subplot
        title2 (str): Title for the second subplot
    """
    # Read both CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Process data for both datasets
    def process_data(df):
        # Group by cause and cause_type, then sum the counts
        aggregated = df.groupby(['cause', 'cause_type'])['count'].sum().reset_index()
        
        primary_df = aggregated[aggregated['cause_type'] == 'primary'].copy()
        secondary_df = aggregated[aggregated['cause_type'] == 'secondary'].copy()
        
        primary_counts = dict(zip(primary_df['cause'], primary_df['count']))
        secondary_counts = dict(zip(secondary_df['cause'], secondary_df['count']))
        
        all_causes = set(primary_counts.keys()) | set(secondary_counts.keys())
        sorted_causes = sorted(all_causes, key=lambda x: primary_counts.get(x, 0), reverse=True)
        
        primary_values = [primary_counts.get(cause, 0) for cause in sorted_causes]
        secondary_values = [secondary_counts.get(cause, 0) for cause in sorted_causes]
        
        return sorted_causes, primary_values, secondary_values
    
    causes1, primary1, secondary1 = process_data(df1)
    causes2, primary2, secondary2 = process_data(df2)
    
    # Find the maximum value across both datasets to set consistent y-axis
    max_value = max(max(primary1 + secondary1), max(primary2 + secondary2))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot first dataset
    x1 = np.arange(len(causes1))
    width = 0.35
    
    bars1_1 = ax1.bar(x1 - width/2, primary1, width, label='Primary', alpha=0.8, color='#1f77b4')
    bars1_2 = ax1.bar(x1 + width/2, secondary1, width, label='Secondary', alpha=0.8, color='#ff7f0e')
    
    ax1.set_xlabel('Causes')
    ax1.set_ylabel('Count')
    ax1.set_title(title1)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(causes1, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max_value * 1.1)  # Add 10% padding above max value
    
    # Plot second dataset
    x2 = np.arange(len(causes2))
    
    bars2_1 = ax2.bar(x2 - width/2, primary2, width, label='Primary', alpha=0.8, color='#1f77b4')
    bars2_2 = ax2.bar(x2 + width/2, secondary2, width, label='Secondary', alpha=0.8, color='#ff7f0e')
    
    ax2.set_xlabel('Causes')
    ax2.set_ylabel('Count')
    ax2.set_title(title2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(causes2, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max_value * 1.1)  # Add 10% padding above max value
    
    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
    
    add_value_labels(ax1, bars1_1)
    add_value_labels(ax1, bars1_2)
    add_value_labels(ax2, bars2_1)
    add_value_labels(ax2, bars2_2)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example usage of the functions
    print("Example usage:")
    print("1. Single plot:")
    plot_cause_statistics('cause_statistics_summary-initial.csv', 'Initial Results')
    print("\n2. Comparison plot:")
    plot_compare_cause_statistics('cause_statistics_summary-initial.csv', 'cause_statistics_summary-improved.csv', 'Initial', 'Improved')
