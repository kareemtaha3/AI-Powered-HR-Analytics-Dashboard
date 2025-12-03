"""
Performance Classification Model Inference Module

This module provides functions to predict employee performance ratings
based on job satisfaction, environment satisfaction, work-life balance,
monthly income, years at company, education, and training times.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# Default model path - resolve from this file's location
_base_dir = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _base_dir / "Models" / "gradient_boosting_performance_classification_model.pkl"
DEFAULT_SCALER_PATH = _base_dir / "Scalers" / "scaler.pkl"

# Required input columns (in order for StandardScaler)
REQUIRED_COLUMNS = [
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'MonthlyIncome',
    'YearsAtCompany',
    'Education',
    'TrainingTimesLastYear'
]

# Performance rating labels
PERFORMANCE_LABELS = {
    2: "Below Average",
    3: "Average",
    4: "Above Average"
}


def get_model_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return resolved path to model file."""
    if path is None:
        return DEFAULT_MODEL_PATH
    return Path(path)


def get_scaler_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return resolved path to scaler file."""
    if path is None:
        return DEFAULT_SCALER_PATH
    return Path(path)


def load_model(model_path: Optional[Union[str, Path]] = None):
    """Load and return the trained model from disk."""
    p = get_model_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def load_scaler(scaler_path: Optional[Union[str, Path]] = None):
    """Load and return the fitted scaler from disk."""
    p = get_scaler_path(scaler_path)
    if not p.exists():
        raise FileNotFoundError(f"Scaler file not found: {p}")
    return joblib.load(p)


def _to_dataframe(input_data: Any) -> pd.DataFrame:
    """Convert input data to DataFrame."""
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    if isinstance(input_data, list) and input_data and isinstance(input_data[0], dict):
        return pd.DataFrame(input_data)
    if isinstance(input_data, pd.DataFrame):
        return input_data
    if isinstance(input_data, pd.Series):
        return input_data.to_frame().T
    raise ValueError("Input must be dict, list of dicts, or pandas DataFrame/Series")


def validate_input(input_data: Any) -> pd.DataFrame:
    """Validate and enforce required columns for model input.
    
    Returns:
    - DataFrame with validated columns in correct order
    
    Raises:
    - ValueError: if required columns are missing
    """
    df = _to_dataframe(input_data)
    
    # Check for missing columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}\n"
            f"Required columns: {REQUIRED_COLUMNS}\n"
            f"Provided columns: {list(df.columns)}"
        )
    
    # Return DataFrame with only required columns in correct order
    return df[REQUIRED_COLUMNS].copy()


def get_input_schema() -> Dict[str, Any]:
    """Return the input schema describing required columns.
    
    Returns:
    - Dictionary with schema information
    """
    return {
        "required_columns": REQUIRED_COLUMNS,
        "column_descriptions": {
            "JobSatisfaction": "Job satisfaction level (1-4 scale)",
            "EnvironmentSatisfaction": "Work environment satisfaction (1-4 scale)",
            "WorkLifeBalance": "Work-life balance rating (1-4 scale)",
            "MonthlyIncome": "Monthly income in currency units",
            "YearsAtCompany": "Number of years at the company",
            "Education": "Education level (1-5 scale: 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)",
            "TrainingTimesLastYear": "Number of trainings attended last year"
        },
        "performance_ratings": PERFORMANCE_LABELS
    }


def example_input() -> Dict[str, Any]:
    """Return an example valid input dictionary.
    
    Returns:
    - Dictionary with example values for all required columns
    """
    return {
        "JobSatisfaction": 3,
        "EnvironmentSatisfaction": 3,
        "WorkLifeBalance": 3,
        "MonthlyIncome": 5000,
        "YearsAtCompany": 5,
        "Education": 3,  # Bachelor's
        "TrainingTimesLastYear": 3
    }


def predict_performance(
    input_data: Any,
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Predict employee performance rating.

    Parameters:
    - input_data: dict, list of dicts, or DataFrame with required columns
    - model_path: optional path to saved model
    - scaler_path: optional path to saved scaler

    Returns dict with keys:
    - "prediction": "Below Average", "Average", or "Above Average"
    - "prediction_label": same as prediction
    - "prediction_numeric": raw rating (2, 3, or 4)
    - "probabilities": dict with probability for each class (if available)
    
    Raises:
    - ValueError: if input is missing required columns
    """
    # Validate input
    X = validate_input(input_data)
    
    # Load scaler and model
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    preds = model.predict(X_scaled)
    
    # Convert numeric predictions to labels
    pred_labels = [PERFORMANCE_LABELS.get(int(p), f"Unknown ({p})") for p in preds]
    
    # For single prediction, return string; for multiple, return list
    if len(pred_labels) == 1:
        prediction_output = pred_labels[0]
        numeric_output = int(preds[0])
    else:
        prediction_output = pred_labels
        numeric_output = preds.tolist()

    result: Dict[str, Any] = {
        "prediction": prediction_output,
        "performance_rating": numeric_output,
        "performance_label": prediction_output,
        "prediction_numeric": numeric_output
    }

    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_scaled)
            classes = model.classes_
            
            if len(proba) == 1:
                # Single prediction
                prob_dict = {
                    PERFORMANCE_LABELS.get(int(cls), f"Unknown ({cls})"): float(prob)
                    for cls, prob in zip(classes, proba[0])
                }
                result["probabilities"] = prob_dict
            else:
                # Multiple predictions
                prob_list = [
                    {
                        PERFORMANCE_LABELS.get(int(cls), f"Unknown ({cls})"): float(prob)
                        for cls, prob in zip(classes, proba_row)
                    }
                    for proba_row in proba
                ]
                result["probabilities"] = prob_list
        except Exception:
            result["probabilities"] = None
    else:
        result["probabilities"] = None

    return result


if __name__ == "__main__":
    # Demo when run as a script
    import json
    
    print("=== Performance Classification Model Helper ===\n")
    
    schema = get_input_schema()
    print("Required columns:", schema["required_columns"])
    print("\nColumn descriptions:")
    for col, desc in schema["column_descriptions"].items():
        print(f"  - {col}: {desc}")
    print("\nPerformance Ratings:", schema["performance_ratings"])
    
    print("\n--- Example input ---")
    example = example_input()
    print(json.dumps(example, indent=2))
    
    print("\n--- Testing validation ---")
    try:
        validated = validate_input(example)
        print(f"✓ Validation passed! Shape: {validated.shape}")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    print("\n--- Example prediction output (mock) ---")
    print("If model is available, output would look like:")
    print(json.dumps({
        "prediction": "Above Average",
        "prediction_label": "Above Average",
        "prediction_numeric": 4,
        "probabilities": {
            "Below Average": 0.05,
            "Average": 0.25,
            "Above Average": 0.70
        }
    }, indent=2))
