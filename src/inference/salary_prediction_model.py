"""
Salary Prediction Model Inference Module

This module provides functions to predict annual salary (ConvertedCompYearly)
for developers based on experience, country, education, employment type, 
developer type, and programming languages.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# Default model path - resolve from this file's location
_base_dir = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _base_dir / "Models" / "best_salary_model_XGBRegressor.pkl"

# Required input columns (features used in training)
REQUIRED_COLUMNS = [
    'YearsCodePro',
    'Country',
    'EdLevel',
    'Employment',
    'DevType_first',
    'Lang_first'
]

NUMERIC_FEATURES = ['YearsCodePro']

CATEGORICAL_FEATURES = [
    'Country',
    'EdLevel',
    'Employment',
    'DevType_first',
    'Lang_first'
]

# Common category values (from Stack Overflow survey data)
COMMON_COUNTRIES = [
    "United States of America",
    "India",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "Canada",
    "France",
    "Brazil",
    "Spain",
    "Netherlands",
    "Australia",
    "Poland",
    "Italy",
    "Sweden",
    "Russian Federation",
    "Switzerland",
    "Egypt"
]

COMMON_EDUCATION_LEVELS = [
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Some college/university study without earning a degree",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "Associate degree (A.A., A.S., etc.)",
    "Professional degree (JD, MD, Ph.D, Ed.D., etc.)",
    "Primary/elementary school",
    "Something else"
]

COMMON_EMPLOYMENT_TYPES = [
    "Employed, full-time",
    "Employed, part-time",
    "Independent contractor, freelancer, or self-employed",
    "Student, full-time",
    "Student, part-time",
    "Not employed, but looking for work",
    "Not employed, and not looking for work",
    "Retired"
]

COMMON_DEV_TYPES = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "Developer, desktop or enterprise applications",
    "DevOps specialist",
    "Engineer, data",
    "Data scientist or machine learning specialist",
    "Database administrator",
    "Designer",
    "System administrator",
    "Engineering manager",
    "Product manager",
    "Educator",
    "Data or business analyst"
]

COMMON_LANGUAGES = [
    "JavaScript",
    "Python",
    "TypeScript",
    "Java",
    "C#",
    "C++",
    "PHP",
    "C",
    "Go",
    "Rust",
    "Kotlin",
    "Ruby",
    "Swift",
    "R",
    "SQL",
    "HTML/CSS",
    "Bash/Shell"
]


def get_model_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return resolved path to model file."""
    if path is None:
        return DEFAULT_MODEL_PATH
    return Path(path)


def load_model(model_path: Optional[Union[str, Path]] = None):
    """Load and return the trained model pipeline from disk."""
    p = get_model_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
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
    - ValueError: if required columns are missing or have invalid values
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
    
    # Validate YearsCodePro is numeric
    if not pd.api.types.is_numeric_dtype(df['YearsCodePro']):
        raise ValueError(
            f"YearsCodePro must be numeric. Got: {df['YearsCodePro'].dtype}"
        )
    
    # Return DataFrame with only required columns in correct order
    return df[REQUIRED_COLUMNS].copy()


def predict_salary(
    input_data: Any,
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Predict annual salary for developers.
    
    Args:
        input_data: dict, list of dicts, or DataFrame with required columns
        model_path: Optional path to model file (uses default if None)
    
    Returns:
        Dictionary with prediction results:
        - predicted_salary: float, predicted annual salary in USD
        - currency: str, always "USD"
        - input_data: dict, validated input used for prediction
    
    Raises:
        ValueError: if input validation fails
        FileNotFoundError: if model file doesn't exist
    
    Example:
        >>> input_data = {
        ...     'YearsCodePro': 3,
        ...     'Country': 'Egypt',
        ...     'EdLevel': "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        ...     'Employment': 'Employed, full-time',
        ...     'DevType_first': 'Developer, full-stack',
        ...     'Lang_first': 'JavaScript'
        ... }
        >>> result = predict_salary(input_data)
        >>> print(result['predicted_salary'])
    """
    # Validate input
    df = validate_input(input_data)
    
    # Load model
    model = load_model(model_path)
    
    # Make prediction
    predictions = model.predict(df)
    
    # Prepare result
    if len(predictions) == 1:
        # Single prediction
        return {
            "predicted_salary": float(predictions[0]),
            "currency": "USD",
            "input_data": df.iloc[0].to_dict()
        }
    else:
        # Batch prediction
        return {
            "predicted_salaries": predictions.tolist(),
            "currency": "USD",
            "count": len(predictions),
            "input_data": df.to_dict(orient='records')
        }


def get_input_schema() -> Dict[str, Any]:
    """Return the expected input schema for the model.
    
    Returns:
        Dictionary describing required columns, types, and valid values
    """
    return {
        "required_columns": REQUIRED_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_info": {
            "YearsCodePro": {
                "type": "numeric",
                "description": "Years of professional coding experience",
                "min": 0,
                "max": 60,
                "example": 3
            },
            "Country": {
                "type": "categorical",
                "description": "Country of residence",
                "common_values": COMMON_COUNTRIES,
                "example": "United States of America"
            },
            "EdLevel": {
                "type": "categorical",
                "description": "Education level",
                "valid_values": COMMON_EDUCATION_LEVELS,
                "example": "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
            },
            "Employment": {
                "type": "categorical",
                "description": "Employment status",
                "valid_values": COMMON_EMPLOYMENT_TYPES,
                "example": "Employed, full-time"
            },
            "DevType_first": {
                "type": "categorical",
                "description": "Primary developer type/role",
                "common_values": COMMON_DEV_TYPES,
                "example": "Developer, full-stack"
            },
            "Lang_first": {
                "type": "categorical",
                "description": "Primary programming language",
                "common_values": COMMON_LANGUAGES,
                "example": "JavaScript"
            }
        }
    }


def example_input() -> Dict[str, Any]:
    """Return an example valid input for the model.
    
    Returns:
        Dictionary with example values for all required columns
    """
    return {
        'YearsCodePro': 3,
        'Country': 'Egypt',
        'EdLevel': "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        'Employment': 'Employed, full-time',
        'DevType_first': 'Developer, full-stack',
        'Lang_first': 'JavaScript'
    }


# Main execution example
if __name__ == "__main__":
    print("Salary Prediction Model - Inference Module")
    print("=" * 60)
    
    # Display schema
    print("\nInput Schema:")
    schema = get_input_schema()
    print(f"Required columns: {schema['required_columns']}")
    
    # Example prediction
    print("\nExample Prediction:")
    example = example_input()
    print(f"Input: {example}")
    
    try:
        result = predict_salary(example)
        print(f"\nPredicted Salary: ${result['predicted_salary']:,.2f} {result['currency']}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Model file not found. Please train the model first using the notebook.")
    except Exception as e:
        print(f"\nError during prediction: {e}")
