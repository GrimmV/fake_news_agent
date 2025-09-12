#!/usr/bin/env python3
"""
Script to generate set-files from experiment annotations.
Processes all experiment files and creates set-files for each annotation type.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetFileGenerator:
    def __init__(self, experiments_dir: str = "observations/experiments_raw"):
        self.experiments_dir = Path(experiments_dir)
        # Structure: {experiment_id: {annotation_name: {label: [scores]}}}
        self.experiments_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.labels = set()
        self.experiment_info = {}  # Store experiment metadata
    
    def load_experiment_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse a single experiment file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load experiment file {file_path}: {e}")
            return []
    
    def extract_annotations_from_experiment(self, experiment_data: List[Dict[str, Any]], label: str, experiment_id: str):
        """Extract annotations from a single experiment and organize by experiment and annotation type."""
        for entry in experiment_data:
            annotations = entry.get('annotations', [])
            for annotation in annotations:
                annotation_name = annotation.get('name')
                score = annotation.get('score')
                
                if annotation_name and score is not None:
                    self.experiments_data[experiment_id][annotation_name][label].append(score)
                    logger.debug(f"Added {annotation_name} score {score} for label {label} in experiment {experiment_id}")
    
    def process_all_experiments(self):
        """Process all experiment files in the experiments directory."""
        logger.info("Starting to process all experiment files...")
        
        # Get all label directories
        if not self.experiments_dir.exists():
            logger.error(f"Experiments directory {self.experiments_dir} does not exist")
            return
        
        label_dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(label_dirs)} label directories: {[d.name for d in label_dirs]}")
        
        # First pass: collect all experiment IDs and labels
        experiment_files = {}  # {experiment_id: {label: file_path}}
        
        for label_dir in label_dirs:
            label = label_dir.name
            self.labels.add(label)
            logger.info(f"Processing label: {label}")
            
            # Get all experiment files for this label
            exp_files = list(label_dir.glob("exp-*.json"))
            logger.info(f"Found {len(exp_files)} experiment files for label {label}")
            
            for exp_file in exp_files:
                # Extract experiment ID from filename (exp-YYYYMMDD_HHMMSS.json)
                exp_id = exp_file.stem.replace('exp-', '')
                
                if exp_id not in experiment_files:
                    experiment_files[exp_id] = {}
                experiment_files[exp_id][label] = exp_file
        
        # Second pass: process each experiment
        for exp_id, label_files in experiment_files.items():
            logger.info(f"Processing experiment: {exp_id}")
            self.experiment_info[exp_id] = {'labels': list(label_files.keys())}
            
            # Process all label files for this experiment
            for label, file_path in label_files.items():
                logger.info(f"Processing file: {file_path.name} for experiment {exp_id}")
                experiment_data = self.load_experiment_file(file_path)
                if experiment_data:
                    self.extract_annotations_from_experiment(experiment_data, label, exp_id)
            
            # Log what we collected for this experiment
            if exp_id in self.experiments_data:
                for annotation_name, label_data in self.experiments_data[exp_id].items():
                    logger.info(f"Experiment {exp_id} - {annotation_name}: {list(label_data.keys())} labels")
        
        logger.info(f"Processed {len(experiment_files)} experiments")
        logger.info(f"Found labels: {sorted(self.labels)}")
        
        # Debug: Print detailed information about what we collected
        for exp_id, exp_data in self.experiments_data.items():
            logger.info(f"=== Experiment {exp_id} ===")
            for annotation_name, label_data in exp_data.items():
                logger.info(f"  {annotation_name}:")
                for label, scores in label_data.items():
                    logger.info(f"    {label}: {len(scores)} scores")
    
    def generate_set_file_content(self, annotation_name: str, data: Dict[str, List[float]]) -> str:
        """Generate Python code content for a set file."""
        lines = [f"{annotation_name} = {{"]
        
        # Sort labels for consistent output
        for label in sorted(data.keys()):
            scores = data[label]
            # Format scores to 2 decimal places for readability
            formatted_scores = [f"{score:.2f}" for score in scores]
            lines.append(f'    "{label}": [')
            lines.append(f"        {', '.join(formatted_scores)}")
            lines.append("    ],")
        
        lines.append("}")
        return "\n".join(lines)
    
    def get_annotation_types(self) -> set:
        """Get all unique annotation types across all experiments."""
        annotation_types = set()
        for exp_data in self.experiments_data.values():
            annotation_types.update(exp_data.keys())
        return annotation_types
    
    def save_set_file(self, annotation_name: str, content: str, experiment_id: str, output_dir: str = "observations/statistics"):
        """Save a set file to the specified directory."""
        # Create experiment-specific directory
        exp_output_dir = Path(output_dir) / f"set_{experiment_id}"
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = exp_output_dir / f"{annotation_name}.py"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved set file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save set file {file_path}: {e}")
    
    def generate_all_set_files(self, output_dir: str = "observations/statistics"):
        """Generate all set files for each experiment and annotation type."""
        logger.info("Generating set files...")
        
        annotation_types = self.get_annotation_types()
        logger.info(f"Found annotation types: {sorted(annotation_types)}")
        
        for experiment_id, exp_data in self.experiments_data.items():
            logger.info(f"Generating set files for experiment: {experiment_id}")
            
            for annotation_name in annotation_types:
                if annotation_name in exp_data:
                    logger.info(f"Generating set file for: {annotation_name} in experiment {experiment_id}")
                    
                    # Check if all labels have data
                    missing_labels = self.labels - set(exp_data[annotation_name].keys())
                    if missing_labels:
                        logger.warning(f"Missing data for labels {missing_labels} in {annotation_name} for experiment {experiment_id}")
                    
                    # Generate content
                    content = self.generate_set_file_content(annotation_name, exp_data[annotation_name])
                    
                    # Save file
                    self.save_set_file(annotation_name, content, experiment_id, output_dir)
                else:
                    logger.warning(f"No data for annotation {annotation_name} in experiment {experiment_id}")
        
        logger.info(f"Generated set files for {len(self.experiments_data)} experiments")
    
    def print_summary(self):
        """Print a summary of the processed data."""
        logger.info("=== PROCESSING SUMMARY ===")
        logger.info(f"Total experiments: {len(self.experiments_data)}")
        logger.info(f"Labels found: {sorted(self.labels)}")
        
        annotation_types = self.get_annotation_types()
        logger.info(f"Annotation types: {sorted(annotation_types)}")
        
        for experiment_id, exp_data in self.experiments_data.items():
            logger.info(f"Experiment {experiment_id}:")
            for annotation_name, data in exp_data.items():
                total_scores = sum(len(scores) for scores in data.values())
                logger.info(f"  {annotation_name}: {total_scores} total scores across {len(data)} labels")
                
                for label, scores in data.items():
                    logger.info(f"    {label}: {len(scores)} scores")
    
    def run(self, output_dir: str = "observations/statistics"):
        """Main method to run the complete set file generation process."""
        logger.info("Starting set file generation process...")
        
        # Process all experiments
        self.process_all_experiments()
        
        # Generate set files
        self.generate_all_set_files(output_dir)
        
        # Print summary
        self.print_summary()
        
        logger.info("Set file generation completed!")


def main():
    """Main function to run the set file generator."""
    generator = SetFileGenerator()
    generator.run()


if __name__ == "__main__":
    main()
