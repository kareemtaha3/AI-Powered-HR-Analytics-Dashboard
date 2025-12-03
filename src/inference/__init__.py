"""Inference helpers package.

Exports convenience names for all model prediction helper functions.
"""

# Attrition Prediction
from .attrition_prediction_model import (
    predict_attrition,
    get_input_schema as get_attrition_schema,
    example_input as example_attrition_input,
)

# Promotion Eligibility
from .promotion_eligibility_model import (
    predict_promotion,
    get_input_schema as get_promotion_schema,
    example_input as example_promotion_input,
)

# Skill Clustering
from .skill_clustring_model import (
    SkillClusteringPredictor,
)

# Performance Classification
from .performance_classification_model import (
    predict_performance,
    get_input_schema as get_performance_schema,
    example_input as example_performance_input,
)

# Salary Prediction
from .salary_prediction_model import (
    predict_salary,
    get_input_schema as get_salary_schema,
    example_input as example_salary_input,
)

# Employee Engagement Clustering
from .employee_engagement_model import (
    predict_engagement_cluster,
    get_input_schema as get_engagement_schema,
    example_input as example_engagement_input,
)

__all__ = [
    # Attrition
    "predict_attrition",
    "get_attrition_schema",
    "example_attrition_input",
    # Promotion
    "predict_promotion",
    "get_promotion_schema",
    "example_promotion_input",
    # Skill Clustering
    "SkillClusteringPredictor",
    # Performance
    "predict_performance",
    "get_performance_schema",
    "example_performance_input",
    # Salary
    "predict_salary",
    "get_salary_schema",
    "example_salary_input",
    # Engagement
    "predict_engagement_cluster",
    "get_engagement_schema",
    "example_engagement_input",
]

