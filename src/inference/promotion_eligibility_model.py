"""
Inference helper for the promotion eligibility prediction model.

Provides functions to load the saved pipeline and make predictions from
Python data structures (dict, DataFrame, numpy arrays).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# Default model path - resolve from this file's location
_base_dir = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _base_dir / "Models" / "hr_promotion_model_full_pipeline.pkl"

REQUIRED_COLUMNS = [
    'age',
    'no_of_trainings',
    'previous_year_rating',
    'length_of_service',
    'awards_won',
    'avg_training_score',
    'department',
    'education',
    'gender',
    'recruitment_channel'
]

NUMERIC_FEATURES = [
    'age',
    'no_of_trainings',
    'previous_year_rating',
    'length_of_service',
    'awards_won',
    'avg_training_score'
]

CATEGORICAL_FEATURES = [
    'department',
    'education',
    'gender',
    'recruitment_channel'
]

CATEGORY_VALUES = {
    'department': ['sales', 'operations', 'it', 'analytics', 'finance', 'hr', 'legal', 'procurement'],
    'education': ['bachelor', 'master', 'phd'],
    'gender': ['m', 'f'],
    'recruitment_channel': ['sourcing', 'referred', 'campus']
}


def get_model_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return a resolved Path to the model file.

    If `path` is None the default model path is used.
    """
    if path is None:
        return DEFAULT_MODEL_PATH
    return Path(path)


def load_model(model_path: Optional[Union[str, Path]] = None):
    """Load and return a joblib model/pipeline from disk.

    Raises FileNotFoundError if the model file does not exist.
    """
    p = get_model_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def _to_2d_array(input_data: Any) -> Any:
    """Convert input data to something the model can accept.

    - dict -> single-row DataFrame
    - list of dicts -> DataFrame
    - pandas Series/DataFrame -> unchanged
    - numpy array -> unchanged
    """
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    if isinstance(input_data, list) and input_data and isinstance(input_data[0], dict):
        return pd.DataFrame(input_data)

    if isinstance(input_data, (pd.DataFrame, pd.Series, np.ndarray)):
        return input_data

    try:
        arr = np.asarray(input_data)
        return arr
    except Exception:
        raise ValueError("Unsupported input_data type for prediction")


def validate_input(input_data: Any) -> pd.DataFrame:
    """Validate and enforce required columns for model input.
    
    Parameters:
    - input_data: dict, list of dicts, or DataFrame
    
    Returns:
    - DataFrame with validated columns in the correct order
    
    Raises:
    - ValueError: if required columns are missing or invalid category values
    """

    df = _to_2d_array(input_data)
    
    if isinstance(df, np.ndarray):
        raise ValueError(
            "Please provide input as a dictionary or DataFrame with named columns.\n"
            f"Required columns: {REQUIRED_COLUMNS}"
        )
    
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}\n"
            f"Required columns: {REQUIRED_COLUMNS}\n"
            f"Provided columns: {list(df.columns)}"
        )
    
    # Make a copy to avoid modifying the original
    df = df[REQUIRED_COLUMNS].copy()
    
    # Normalize categorical features to lowercase for consistency with training data
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            # Convert to string and lowercase to match training data format
            df[cat_col] = df[cat_col].astype(str).str.lower().str.strip()
    

    return df


def get_input_schema() -> Dict[str, Any]:
    """Return the input schema describing required columns and their types.
    
    Returns:
    - Dictionary with schema information
    """
    return {
        "required_columns": REQUIRED_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_values": CATEGORY_VALUES,
        "total_features": len(REQUIRED_COLUMNS)
    }


def example_input() -> Dict[str, Any]:
    """Return an example valid input dictionary.
    
    Returns:
    - Dictionary with example values for all required columns
    """
    return {
        "age": 35,
        "no_of_trainings": 3,
        "previous_year_rating": 3.5,
        "length_of_service": 5,
        "awards_won": 1,
        "avg_training_score": 75,
        "department": "sales",
        "education": "bachelor",
        "gender": "m",
        "recruitment_channel": "referred"
    }


def predict_promotion(input_data: Any, model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load model (if needed) and predict promotion eligibility.

    Parameters:
    - input_data: dict, list of dicts, or pandas DataFrame with required columns
    - model_path: optional path to a saved model; default uses the repository default

    Returns a dict with keys:
    - "prediction": "Promoted" or "Not Promoted" (or list for multiple inputs)
    - "prediction_label": same as prediction (for clarity)
    - "promotion_probability": probability of promotion (0.0 to 1.0, or list for multiple)
    - "prediction_numeric": 0 (Not Promoted) or 1 (Promoted) - raw model output
    
    Raises:
    - ValueError: if input is missing required columns or has invalid values
    """

    X = validate_input(input_data)
    

    model = load_model(model_path)


    preds = model.predict(X)
    

    pred_labels = ["Promoted" if p == 1 else "Not Promoted" for p in preds]
    

    if len(pred_labels) == 1:
        prediction_output = pred_labels[0]
        numeric_output = int(preds[0])
    else:
        prediction_output = pred_labels
        numeric_output = preds.tolist()

    result: Dict[str, Any] = {
        "prediction": prediction_output,
        "prediction_label": prediction_output,
        "prediction_numeric": numeric_output
    }


    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)

            if proba.shape[1] == 2:
                promotion_probs = proba[:, 1]
                if len(promotion_probs) == 1:
                    result["promotion_probability"] = float(promotion_probs[0])
                else:
                    result["promotion_probability"] = promotion_probs.tolist()
            else:
                result["promotion_probability"] = proba.tolist()
        except Exception:
            result["promotion_probability"] = None
    else:
        result["promotion_probability"] = None

    return result


def predict_proba(input_data: Any, model_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """Return prediction probabilities (raises if unavailable).
    
    Parameters:
    - input_data: dict, list of dicts, or DataFrame with required columns
    - model_path: optional path to saved model
    
    Returns:
    - numpy array of probabilities
    """
    X = validate_input(input_data)
    model = load_model(model_path)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba")
    return model.predict_proba(X)


if __name__ == "__main__":
    # Demo when run as a script
    print("=== Promotion Eligibility Prediction Model Helper ===\n")
    
    schema = get_input_schema()
    print("Required columns:", schema["required_columns"])
    print("\nNumeric features:", schema["numeric_features"])
    print("\nCategorical features:", schema["categorical_features"])
    print("\nCategory value constraints:", schema["category_values"])
    
    print("\n--- Example input ---")
    example = example_input()
    import json
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
        "prediction": "Promoted",
        "prediction_label": "Promoted",
        "promotion_probability": 0.78,
        "prediction_numeric": 1
    }, indent=2))
    print("\nOr for Not Promoted:")
    print(json.dumps({
        "prediction": "Not Promoted",
        "prediction_label": "Not Promoted",
        "promotion_probability": 0.32,
        "prediction_numeric": 0
    }, indent=2))

    print("\n--- Testing prediction ---")
    try:
        result = predict_promotion(example)
        print("Predicted label(s):", result['prediction'])
        print("Predicted probability (positive class):", result['promotion_probability'])
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
