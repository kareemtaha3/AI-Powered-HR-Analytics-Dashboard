"""
Employee Engagement Clustering Model Inference Module

This module provides functions to assign employees to engagement clusters
based on job satisfaction, work environment, income, tenure, and performance factors.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# Default model path - resolve from this file's location
_base_dir = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _base_dir / "Models" / "employee_engagement_kmeans_model.pkl"
DEFAULT_SCALER_PATH = _base_dir / "Models" / "employee_engagement_scaler.pkl"

# Required input columns (features used in clustering)
REQUIRED_COLUMNS = [
    'Age',
    'MonthlyIncome',
    'JobLevel',
    'YearsAtCompany',
    'TotalWorkingYears',
    'PromotionRate',
    'Department_enc',
    'JobRole_enc',
    'JobInvolvement',
    'StockOptionLevel'
]

NUMERIC_FEATURES = [
    'Age',
    'MonthlyIncome',
    'JobLevel',
    'YearsAtCompany',
    'TotalWorkingYears',
    'PromotionRate',
    'JobInvolvement',
    'StockOptionLevel'
]

CATEGORICAL_FEATURES = [
    'Department_enc',
    'JobRole_enc'
]

# Cluster names and descriptions (from notebook analysis)
CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Highly Engaged & Loyal",
        "description": "Senior employees with high job satisfaction, strong loyalty, and excellent compensation efficiency",
        "characteristics": [
            "High job satisfaction and involvement",
            "Long tenure and loyalty",
            "Strong performance and promotion history",
            "Competitive compensation"
        ]
    },
    1: {
        "name": "Actively Disengaged",
        "description": "Employees showing signs of disengagement with lower satisfaction and involvement",
        "characteristics": [
            "Lower job satisfaction and involvement",
            "Limited career progression",
            "May be at risk of attrition",
            "Need development and engagement initiatives"
        ]
    },
    2: {
        "name": "Moderately Engaged",
        "description": "Mid-level employees with balanced engagement and steady career progression",
        "characteristics": [
            "Average job satisfaction",
            "Moderate tenure and experience",
            "Stable performance",
            "Room for growth and development"
        ]
    },
    3: {
        "name": "New & Enthusiastic",
        "description": "Early-career employees with high enthusiasm and learning potential",
        "characteristics": [
            "Recently joined the company",
            "High learning agility",
            "Strong potential for growth",
            "Need mentoring and development"
        ]
    }
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
    """Load and return the trained clustering model from disk."""
    p = get_model_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def load_scaler(scaler_path: Optional[Union[str, Path]] = None):
    """Load and return the fitted scaler from disk."""
    p = get_scaler_path(scaler_path)
    if not p.exists():
        # Scaler is optional, return None if not found
        return None
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
    
    # Validate numeric features
    for col in NUMERIC_FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric. Got: {df[col].dtype}")
    
    # Return DataFrame with only required columns in correct order
    return df[REQUIRED_COLUMNS].copy()


def predict_engagement_cluster(
    input_data: Any,
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Predict employee engagement cluster.
    
    Args:
        input_data: dict, list of dicts, or DataFrame with required columns
        model_path: Optional path to model file (uses default if None)
        scaler_path: Optional path to scaler file (uses default if None)
    
    Returns:
        Dictionary with prediction results:
        - cluster_id: int, cluster assignment (0-3)
        - cluster_name: str, human-readable cluster name
        - cluster_description: str, description of the cluster
        - characteristics: list, key characteristics of the cluster
        - input_data: dict, validated input used for prediction
    
    Raises:
        ValueError: if input validation fails
        FileNotFoundError: if model file doesn't exist
    
    Example:
        >>> input_data = {
        ...     'Age': 35,
        ...     'MonthlyIncome': 5000,
        ...     'JobLevel': 2,
        ...     'YearsAtCompany': 5,
        ...     'TotalWorkingYears': 10,
        ...     'PromotionRate': 0.2,
        ...     'Department_enc': 1,
        ...     'JobRole_enc': 3,
        ...     'JobInvolvement': 3,
        ...     'StockOptionLevel': 1
        ... }
        >>> result = predict_engagement_cluster(input_data)
        >>> print(result['cluster_name'])
    """
    # Validate input
    df = validate_input(input_data)
    
    # Load model
    model = load_model(model_path)
    
    # Load scaler (optional)
    scaler = load_scaler(scaler_path)
    if scaler is not None:
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns
        )
    else:
        df_scaled = df
    
    # Make prediction
    cluster_ids = model.predict(df_scaled)
    
    # Prepare result
    if len(cluster_ids) == 1:
        # Single prediction
        cluster_id = int(cluster_ids[0])
        cluster_info = CLUSTER_DESCRIPTIONS.get(cluster_id, {
            "name": f"Cluster {cluster_id}",
            "description": "Engagement profile",
            "characteristics": []
        })
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_info["name"],
            "cluster_description": cluster_info["description"],
            "characteristics": cluster_info["characteristics"],
            "input_data": df.iloc[0].to_dict()
        }
    else:
        # Batch prediction
        results = []
        for i, cluster_id in enumerate(cluster_ids):
            cluster_id = int(cluster_id)
            cluster_info = CLUSTER_DESCRIPTIONS.get(cluster_id, {
                "name": f"Cluster {cluster_id}",
                "description": "Engagement profile",
                "characteristics": []
            })
            results.append({
                "cluster_id": cluster_id,
                "cluster_name": cluster_info["name"],
                "cluster_description": cluster_info["description"],
                "characteristics": cluster_info["characteristics"],
                "input_data": df.iloc[i].to_dict()
            })
        return {
            "predictions": results,
            "count": len(results)
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
            "Age": {
                "type": "numeric",
                "description": "Employee age in years",
                "min": 18,
                "max": 65,
                "example": 35
            },
            "MonthlyIncome": {
                "type": "numeric",
                "description": "Monthly income in USD",
                "min": 1000,
                "max": 20000,
                "example": 5000
            },
            "JobLevel": {
                "type": "numeric",
                "description": "Job level (1-5)",
                "min": 1,
                "max": 5,
                "example": 2
            },
            "YearsAtCompany": {
                "type": "numeric",
                "description": "Years at current company",
                "min": 0,
                "max": 40,
                "example": 5
            },
            "TotalWorkingYears": {
                "type": "numeric",
                "description": "Total years of work experience",
                "min": 0,
                "max": 50,
                "example": 10
            },
            "PromotionRate": {
                "type": "numeric",
                "description": "Promotion rate (0-1)",
                "min": 0,
                "max": 1,
                "example": 0.2
            },
            "Department_enc": {
                "type": "categorical_encoded",
                "description": "Department (encoded as integer)",
                "example": 1
            },
            "JobRole_enc": {
                "type": "categorical_encoded",
                "description": "Job role (encoded as integer)",
                "example": 3
            },
            "JobInvolvement": {
                "type": "numeric",
                "description": "Job involvement level (1-4)",
                "min": 1,
                "max": 4,
                "example": 3
            },
            "StockOptionLevel": {
                "type": "numeric",
                "description": "Stock option level (0-3)",
                "min": 0,
                "max": 3,
                "example": 1
            }
        }
    }


def example_input() -> Dict[str, Any]:
    """Return an example valid input for the model.
    
    Returns:
        Dictionary with example values for all required columns
    """
    return {
        'Age': 35,
        'MonthlyIncome': 5000,
        'JobLevel': 2,
        'YearsAtCompany': 5,
        'TotalWorkingYears': 10,
        'PromotionRate': 0.2,
        'Department_enc': 1,
        'JobRole_enc': 3,
        'JobInvolvement': 3,
        'StockOptionLevel': 1
    }


# Main execution example
if __name__ == "__main__":
    print("Employee Engagement Clustering Model - Inference Module")
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
        result = predict_engagement_cluster(example)
        print(f"\nCluster: {result['cluster_name']}")
        print(f"Description: {result['cluster_description']}")
        print(f"Characteristics:")
        for char in result['characteristics']:
            print(f"  - {char}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Model file not found. Please train the model first using the notebook.")
    except Exception as e:
        print(f"\nError during prediction: {e}")
