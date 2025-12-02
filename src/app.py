from inference import predict_attrition, get_model_path

# single sample as dictionary (keys must match the model's expected columns)
sample = {
    "Age": 35,
    "DistanceFromHome": 10,
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 2,
    "joblevel": 2,
    "jobRole": "Sales Executive",
    "jobSatisfaction": 3,
    "maritalStatus": "Married",
    "monthlyIncome": 5000,
    "MonthlyRate": 10000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "percentSalaryHike": 11,
    "RelationshipSatisfaction": 2,
    "stockOptionLevel": 0,
    "TotalWrokingYears": 10,
    "worklifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
}

result = predict_attrition(sample)  # uses default model path
print('Predicted label(s):', result['prediction'])
print('Predicted probability (positive class):', result['probability'])
# if you want the first sample:
pred_label = result['prediction'][0]
prob_pos = None if result['probability'] is None else result['probability'][0]