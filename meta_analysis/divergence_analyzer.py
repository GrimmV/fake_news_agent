"""
Divergence Analysis Module

This module processes input strings using OpenAI LLM with the divergence analysis
prompt and Pydantic response model. It analyzes why LLM trust judgements diverge
from correct ML model predictions.
"""

import os
import sys
from typing import Optional, Union
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from llm.llm import GPTModel
from meta_analysis.prompts.divergence_analysis import divergence_analysis_prompt
from meta_analysis.response_models.DivergenceAnalysis import DivergenceAnalysis
from meta_analysis.examples.example1 import example1
from dotenv import load_dotenv

load_dotenv(override=True)


class DivergenceAnalyzer:
    """Analyzes LLM-ML divergences using OpenAI with structured output."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the divergence analyzer.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model_name: OpenAI model to use for analysis.
        """
        if api_key is None:
            api_key = os.getenv("API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        self.model = GPTModel(model_name=model_name, key=api_key)
        self.prompt_template = divergence_analysis_prompt
    
    def analyze_trace(self, trace: str, case_type: str, label: int, prediction: int, trustscore: int) -> DivergenceAnalysis:
        """
        Analyze a trace for divergence between LLM and ML model predictions.
        
        Args:
            trace: The trace data to analyze (string format).
            case_type: The type of case to analyze.
            label: The label of the case.
            prediction: The prediction of the case.
            trustscore: The trustscore of the case.
            
        Returns:
            DivergenceAnalysis: Structured analysis of the divergence.
        """
        # Format the prompt with the trace data
        formatted_prompt = self.prompt_template(trace=trace, case_type=case_type, label=label, prediction=prediction, trustscore=trustscore)
        
        # Generate analysis using the LLM
        response = self.model.generate(
            prompt=formatted_prompt,
            response_model=DivergenceAnalysis,
            system_message="You are an expert analyst of LLM behavior in hybrid ML + XAI pipelines.",
            max_retries=3
        )
        
        return response.model_dump()
    
    def analyze_from_json(self, json_data: dict) -> DivergenceAnalysis:
        """
        Analyze trace data from a JSON object.
        
        Args:
            json_data: JSON object containing trace data.
            
        Returns:
            DivergenceAnalysis: Structured analysis of the divergence.
        """
        import json
        # Convert JSON to string format for the prompt
        trace_data = json.dumps(json_data, indent=2)
        return self.analyze_trace(trace_data)
    
    def print_analysis(self, analysis: DivergenceAnalysis):
        """Pretty print the analysis results."""
        print("\n" + "="*60)
        print("DIVERGENCE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nPrimary Cause: {analysis.primary_cause.value}")
        
        if analysis.secondary_causes:
            print(f"\nSecondary Causes:")
            for cause in analysis.secondary_causes:
                print(f"  - {cause.value}")
        
        print(f"\nEvidence ({len(analysis.factors)} points):")
        for i, factor in enumerate(analysis.factors, 1):
            print(f"\n{i}. {factor.explanation}")
            print(f"   Reference: {factor.reference}")
            print(f"   Associated Divergence Type: {factor.associated_divergence_type.value}")
        
        print(f"\nMeta Comment:")
        print(f"{analysis.meta_comment}")
        print("\n" + "="*60)


if __name__ == "__main__":
    analyzer = DivergenceAnalyzer(model_name="gpt-5-2025-08-07")
    analysis = analyzer.analyze_trace(example1)
    analyzer.print_analysis(analysis)