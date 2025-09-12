# Divergence Analysis Script

This directory contains tools for analyzing divergences between LLM trust judgements and ML model predictions using OpenAI's API with structured output via Pydantic models.

## Files

- `divergence_analyzer.py` - Main script for processing input strings with OpenAI LLM
- `example_usage.py` - Examples showing different ways to use the analyzer
- `config.py` - Configuration settings
- `prompts/divergence_analysis.py` - The prompt template for divergence analysis
- `response_models/DivergenceAnalysis.py` - Pydantic model for structured responses

## Setup

1. Install dependencies:
   ```bash
   pip install openai instructor pydantic
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Command Line Interface

```bash
# Analyze trace data from string
python divergence_analyzer.py "your trace data here"

# Analyze from file
python divergence_analyzer.py --input-file trace_data.json

# Interactive mode
python divergence_analyzer.py --interactive

# Use different model
python divergence_analyzer.py --model gpt-4o "trace data"

# Save results to file
python divergence_analyzer.py --output results.json "trace data"

# Quiet mode (JSON output only)
python divergence_analyzer.py --quiet "trace data"
```

### Python API

```python
from divergence_analyzer import DivergenceAnalyzer

# Initialize analyzer
analyzer = DivergenceAnalyzer()

# Analyze trace data
analysis = analyzer.analyze_trace("your trace data here")

# Print results
analyzer.print_analysis(analysis)

# Analyze from file
analysis = analyzer.analyze_from_file("trace_data.json")

# Analyze from JSON object
trace_data = {"ml_prediction": "true", "llm_trust_judgement": "distrust", ...}
analysis = analyzer.analyze_from_json(trace_data)
```

## Input Format

The script expects trace data that contains information about:
- ML model prediction
- LLM trust judgement
- Global metrics (accuracy, F1 score, etc.)
- Individual feature importance
- Explanations provided

Example trace data:
```json
{
    "ml_prediction": "true",
    "llm_trust_judgement": "distrust",
    "global_metrics": {
        "accuracy": 0.75,
        "f1_score": 0.72
    },
    "individual_feature_importance": {
        "summary": "High importance on sensational language features"
    },
    "explanation": "The model shows concerning patterns..."
}
```

## Output Format

The script returns a structured `DivergenceAnalysis` object with:
- `primary_cause`: Main reason for divergence
- `secondary_causes`: Additional contributing factors
- `factors`: Evidence points with explanations and references
- `meta_comment`: Summary connecting factors to evidence

## Divergence Types

The analysis identifies these types of divergences:
- `performance_baseline`: Distrust due to weak global metrics
- `feature_interpretation_bias`: Mis/over-weighting risky local features
- `explanation_framing`: Alarmist/critical wording in explanations
- `label_trust_mismatch`: Confuses label meaning with trust
- `overgeneralization_from_dataset_statistics`: Misapplies aggregate statistics
- `meta_performance_overweighting`: Demands unrealistically high certainty
- `content_model_confusion`: Judges the claim itself, not model trustworthiness
- `other`: Residual causes

## Examples

See `example_usage.py` for comprehensive examples of different usage patterns.

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`
