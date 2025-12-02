import joblib
import pandas as pd

# Load the saved model (contains preprocessor + classifier)
loaded_model = joblib.load("hr_promotion_model_full_pipeline.pkl")

new_data = pd.DataFrame(
    {
        "department": ["Sales & Marketing"],
        "education": ["Master's & above"],
        "gender": ["f"],
        "recruitment_channel": ["sourcing"],
        "no_of_trainings": [1],
        "age": [35],
        "previous_year_rating": [5],
        "length_of_service": [8],
        "awards_won": [0],
        "avg_training_score": [49],
    }
)

def predict(input_data: pd.DataFrame):
    """
    Predict promotion eligibility for new employee data.

    Args:
        input_data (pd.DataFrame): New employee data in raw format.

    Returns:
        tuple: (prediction, probability)
    """
    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data)[:, 1]
    return prediction, probability


if __name__ == "__main__":
    # Ex.
    print(predict(new_data))
