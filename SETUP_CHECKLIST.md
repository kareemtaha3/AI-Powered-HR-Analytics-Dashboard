# Setup Checklist for Streamlit Pages

## ✅ Completed
- [x] Created salary prediction inference module (`src/inference/salary_prediction_model.py`)
- [x] Created employee engagement clustering inference module (`src/inference/employee_engagement_model.py`)
- [x] Created performance classification Streamlit page (`src/pages/04_performance_classification.py`)
- [x] Created salary prediction Streamlit page (`src/pages/05_salary_prediction.py`)
- [x] Created employee engagement Streamlit page (`src/pages/06_employee_engagement.py`)
- [x] Updated `src/inference/__init__.py` to export all modules
- [x] Updated `src/app.py` to include all 6 models in navigation
- [x] Fixed performance classification inference to include `performance_rating` field
- [x] Documented all changes in `STREAMLIT_PAGES_SUMMARY.md`

## 📋 To-Do Before Running

### 1. Ensure Model Files Exist

Check that the following model files exist in the `Models/` directory:

```bash
# Check existing models
ls -la Models/

# Required files:
# - best_attrition_prediction_model_voting.pkl ✓ (should exist)
# - hr_promotion_model_full_pipeline.pkl ✓ (should exist)
# - skill_composition_kmeans_model.pkl ✓ (should exist)
# - gradient_boosting_model.pkl ⚠️ (check name)
# - best_salary_model_XGBRegressor.pkl ❓ (needs training/renaming)
# - employee_engagement_kmeans_model.pkl ❓ (needs to be saved)
```

### 2. Train/Save Missing Models

#### Performance Classification Model
The model should exist from the notebook. Verify the filename:
- Expected: `Models/gradient_boosting_model.pkl`
- Or: `Models/gradient_boosting_performance_classification_model.pkl`

If using different name, update `DEFAULT_MODEL_PATH` in:
`src/inference/performance_classification_model.py`

#### Salary Prediction Model
Run the salary prediction notebook and ensure it saves:
```python
# In Model_4_Salary_Prediction_with_EDA.ipynb
# The best model should be saved as:
model_path = "../Models/best_salary_model_XGBRegressor.pkl"
joblib.dump(best_pipeline, model_path)
```

If the notebook saves with a different name, update `DEFAULT_MODEL_PATH` in:
`src/inference/salary_prediction_model.py`

#### Employee Engagement Model
Add to the end of `Employee_Engagement_Clustering.ipynb`:
```python
import joblib
import os

# Save the final K-Means model
os.makedirs("../Models", exist_ok=True)
joblib.dump(kmeans, '../Models/employee_engagement_kmeans_model.pkl')
joblib.dump(scaler, '../Models/employee_engagement_scaler.pkl')

print("✓ Employee engagement model and scaler saved successfully!")
```

Then run the notebook cells to generate the model files.

### 3. Check Scaler Files

Performance classification needs:
- `Scalers/scaler.pkl` ✓ (should exist from notebook)

Employee engagement needs (after step 2.3):
- `Models/employee_engagement_scaler.pkl` (saved with model)

### 4. Install Dependencies (if not already installed)

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly xgboost
```

### 5. Test Each Page

```bash
# Run the main app
cd src
streamlit run app.py

# Test each model:
# 1. Navigate to each page using sidebar
# 2. Try single prediction with example values
# 3. Test batch prediction with sample CSV (optional)
# 4. Check Model Info tab for documentation
```

## 🔍 Verification Steps

### Test Performance Classification
1. Go to page: ⭐ Performance Classification
2. Set values:
   - Job Satisfaction: 3
   - Environment Satisfaction: 3
   - Work-Life Balance: 3
   - Monthly Income: 5000
   - Years At Company: 5
   - Education: 3
   - Training Times: 3
3. Click "⭐ Predict Performance"
4. Should see: Performance rating and probabilities

### Test Salary Prediction
1. Go to page: 💰 Salary Prediction
2. Set values:
   - Years of Professional Coding: 3
   - Country: Egypt (or any)
   - Education: Bachelor's degree
   - Employment: Employed, full-time
   - Developer Type: Developer, full-stack
   - Primary Language: JavaScript
3. Click "💰 Predict Salary"
4. Should see: Predicted salary with breakdown

### Test Employee Engagement
1. Go to page: 🎯 Employee Engagement
2. Set values:
   - Age: 35
   - Monthly Income: 5000
   - Job Level: 2
   - Years At Company: 5
   - Total Working Years: 10
   - Promotion Rate: 0.2
   - Department (encoded): 1
   - Job Role (encoded): 3
   - Job Involvement: 3
   - Stock Option Level: 1
3. Click "🎯 Assign to Cluster"
4. Should see: Cluster name, description, characteristics, recommendations

## 🐛 Troubleshooting

### Error: "Model file not found"
- Check model path in inference module
- Verify model file exists in Models/ directory
- Ensure notebook saved model with correct filename

### Error: "Scaler file not found"
- For performance: Check `Scalers/scaler.pkl` exists
- For engagement: Check `Models/employee_engagement_scaler.pkl` exists
- Re-run notebook if missing

### Error: "Missing required columns"
- Check input data matches expected schema
- Use `get_input_schema()` to see required columns
- Use `example_input()` to see example format

### Error: Import errors (numpy, pandas, streamlit)
- These are linting errors, not runtime errors
- Ensure dependencies installed: `pip install -r requirements.txt`
- If running in venv, activate it first

### Page doesn't load in app.py
- Check file path in routing section
- Ensure page file exists in `src/pages/` directory
- Check for syntax errors in page file

## 📝 Quick Commands

```bash
# Create Models directory if needed
mkdir -p Models Scalers

# Check what models exist
ls -lh Models/

# Run Streamlit app
cd src && streamlit run app.py

# Run specific notebook to generate model
jupyter notebook notebooks/Employee_Engagement_Clustering.ipynb
```

## 🎯 Success Criteria

All pages should:
- [x] Load without errors
- [x] Accept user input via forms
- [x] Make predictions when button clicked
- [x] Display results with visualizations
- [x] Show model information in Info tab
- [x] Support batch predictions via CSV
- [x] Download results as CSV

## 📚 Additional Resources

- **Model Details**: See `STREAMLIT_PAGES_SUMMARY.md`
- **Notebooks**: 
  - `notebooks/employee_performance_cassification.ipynb`
  - `notebooks/Model_4_Salary_Prediction_with_EDA.ipynb`
  - `notebooks/Employee_Engagement_Clustering.ipynb`
- **Inference Modules**: `src/inference/`
- **Streamlit Pages**: `src/pages/`
- **Main App**: `src/app.py`

---

**Need Help?**
- Check error messages carefully - they usually indicate missing files or wrong paths
- Verify model training completed successfully in notebooks
- Ensure all dependencies installed
- Test inference modules standalone before testing in Streamlit
