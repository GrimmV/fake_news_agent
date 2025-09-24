#!/usr/bin/env python3
"""
Script to retrieve experiments from Arize Phoenix server and organize them by label.
"""

import requests
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

version = "v1.1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhoenixDataRetriever:
    def __init__(self, base_url: str = "http://localhost:6006"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Retrieve all datasets from Phoenix server."""
        try:
            response = self.session.get(f"{self.base_url}/v1/datasets")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve datasets: {e}")
            return []
    
    def filter_datasets_by_pattern(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter datasets by name pattern and extract label."""
        filtered_datasets = []
        # Create a more specific regex that captures everything between the pattern parts
        # This handles labels with underscores by matching everything between "patterns_" and "_v1"
        pattern_regex = re.compile(fr'structured_experiment_inputs_patterns_(.+)_{version}')
        
        for dataset in datasets["data"]:
            name = dataset.get('name', '')
            match = pattern_regex.match(name)
            if match:
                label = match.group(1)
                dataset['extracted_label'] = label
                filtered_datasets.append(dataset)
                logger.info(f"Found matching dataset: {name} -> label: {label}")
        
        return filtered_datasets
    
    def get_experiments_for_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieve experiments for a specific dataset."""
        try:
            response = self.session.get(f"{self.base_url}/v1/datasets/{dataset_id}/experiments")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve experiments for dataset {dataset_id}: {e}")
            return []
    
    def get_experiment_json(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve full experiment data in JSON format."""
        try:
            response = self.session.get(f"{self.base_url}/v1/experiments/{experiment_id}/json")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return {}
    
    def create_label_directory(self, label: str) -> Path:
        """Create directory for the label if it doesn't exist."""
        label_dir = Path(f"observations/experiments_raw_{version}") / label
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir
    
    def format_created_at(self, created_at: str) -> str:
        """Format created_at timestamp for filename."""
        try:
            # Parse the timestamp and format it for filename
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            return dt.strftime("%Y%m%d_%H%M%S")
        except Exception as e:
            logger.warning(f"Failed to parse created_at '{created_at}': {e}")
            # Fallback to a safe filename
            safe_timestamp = created_at.replace(':', '-').replace('T', '_').replace('Z', '').replace('.', '_')
            return safe_timestamp
    
    def save_experiment(self, experiment_data: Dict[str, Any], label: str, created_at: str) -> str:
        """Save experiment data to file."""
        label_dir = self.create_label_directory(label)
        timestamp = self.format_created_at(created_at)
        filename = f"exp-{timestamp}.json"
        filepath = label_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved experiment to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save experiment to {filepath}: {e}")
            return ""
    
    def save_experiment_with_number(self, experiment_data: Dict[str, Any], label: str, exp_number: int) -> str:
        """Save experiment data to file with sequential numbering."""
        label_dir = self.create_label_directory(label)
        filename = f"exp-{exp_number}.json"
        filepath = label_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved experiment to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save experiment to {filepath}: {e}")
            return ""
    
    def retrieve_all_experiments(self, pattern: str = "structured_experiment_inputs_patterns_{label}_"):
        """Main method to retrieve and organize all experiments."""
        logger.info("Starting experiment retrieval process...")
        
        pattern = pattern + version
        
        # Step 1: Get all datasets
        logger.info("Retrieving datasets...")
        datasets = self.get_datasets()
        if not datasets:
            logger.error("No datasets found or failed to retrieve datasets")
            return
        
        logger.info(f"Found {len(datasets)} total datasets")
        
        # Step 2: Filter datasets by pattern
        logger.info(f"Filtering datasets by pattern: {pattern}")
        filtered_datasets = self.filter_datasets_by_pattern(datasets)
        
        if not filtered_datasets:
            logger.warning(f"No datasets found matching pattern: {pattern}")
            return
        
        logger.info(f"Found {len(filtered_datasets)} matching datasets")
        
        # Step 3: Process each dataset separately and number experiments per label
        total_experiments = 0
        
        for dataset in filtered_datasets:
            dataset_id = dataset['id']
            label = dataset['extracted_label']
            dataset_name = dataset['name']
            
            logger.info(f"Processing dataset: {dataset_name} (ID: {dataset_id}, Label: {label})")
            
            # Get experiments for this dataset
            experiments = self.get_experiments_for_dataset(dataset_id)
            if not experiments:
                logger.warning(f"No experiments found for dataset {dataset_name}")
                continue
            
            logger.info(f"Found {len(experiments)} experiments for dataset {dataset_name}")
            
            # Collect experiments for this label
            label_experiments = []
            for experiment in experiments["data"]:
                experiment_id = experiment['id']
                created_at = experiment.get('created_at', '')
                
                logger.info(f"Retrieving experiment {experiment_id}...")
                
                # Get full experiment data
                experiment_data = self.get_experiment_json(experiment_id)
                if not experiment_data:
                    logger.warning(f"Failed to retrieve data for experiment {experiment_id}")
                    continue
                
                label_experiments.append((experiment_data, created_at, experiment_id))
            
            # Sort experiments for this label by created_at timestamp
            logger.info(f"Sorting {len(label_experiments)} experiments for label {label} by creation time...")
            label_experiments.sort(key=lambda x: x[1])  # Sort by created_at (index 1)
            
            # Save experiments with sequential numbering for this label
            for exp_index, (experiment_data, created_at, experiment_id) in enumerate(label_experiments):
                exp_number = exp_index + 1  # Start from 1 instead of 0
                filepath = self.save_experiment_with_number(experiment_data, label, exp_number)
                if filepath:
                    total_experiments += 1
                    logger.info(f"Saved experiment {exp_number} for label {label}")
        
        logger.info(f"Retrieval process completed. Total experiments saved: {total_experiments}")


def main():
    """Main function to run the data retrieval process."""
    retriever = PhoenixDataRetriever()
    retriever.retrieve_all_experiments()


if __name__ == "__main__":
    main()
