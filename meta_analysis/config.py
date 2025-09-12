"""
Configuration settings for the divergence analysis system.
"""

import os
from pathlib import Path

# Default model settings
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1500

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# File paths
META_ANALYSIS_DIR = Path(__file__).parent
PROMPTS_DIR = META_ANALYSIS_DIR / "prompts"
RESPONSE_MODELS_DIR = META_ANALYSIS_DIR / "response_models"

# Output settings
DEFAULT_OUTPUT_DIR = META_ANALYSIS_DIR / "outputs"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validation settings
MAX_RETRIES = 3
VALIDATION_TIMEOUT = 30  # seconds
