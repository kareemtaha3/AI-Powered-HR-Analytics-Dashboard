# 🏢 AI-Powered HR Analytics Dashboard

A comprehensive machine learning-powered HR analytics platform built with Python and Streamlit, featuring 6 predictive models for employee management, retention, and workforce planning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project is an end-to-end machine learning application designed to help HR professionals and business leaders make data-driven decisions about their workforce. The dashboard integrates six sophisticated ML models covering various aspects of employee lifecycle management, from recruitment to retention.

**Key Benefits:**
- 🎯 **Predictive Analytics**: Forecast attrition, performance, and promotion readiness
- 💰 **Compensation Planning**: Benchmark salaries and plan hiring budgets
- 🔍 **Employee Insights**: Understand engagement levels and skill compositions
- 📊 **Interactive Dashboard**: Real-time predictions with intuitive visualizations
- 📈 **Batch Processing**: Support for analyzing multiple employees simultaneously

---

## ✨ Features

### 🚀 Six ML Models

1. **Employee Attrition Prediction** - Identify at-risk employees
2. **Promotion Eligibility** - Assess promotion readiness
3. **Skill Composition Clustering** - Segment employees by technical skills
4. **Performance Classification** - Predict performance ratings
5. **Salary Prediction** - Estimate competitive compensation
6. **Employee Engagement Clustering** - Analyze engagement levels

### 💻 User-Friendly Interface

- **Single Prediction Mode**: Interactive forms with sliders and dropdowns
- **Batch Prediction Mode**: Upload CSV files for bulk analysis
- **Rich Visualizations**: Gauges, charts, probability distributions
- **Model Documentation**: Built-in schema and usage instructions

### 📊 Advanced Analytics

- Prediction probabilities and confidence scores
- Feature importance and model interpretability
- Cluster descriptions and recommendations
- Interactive data exploration

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │ Models   │  │  Batch   │  │  Info    │   │
│  │   Page   │  │  (1-6)   │  │ Predict  │  │  Pages   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Inference Layer (Python)                    │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │  Input Validation  │  │  Data Processing   │            │
│  └────────────────────┘  └────────────────────┘            │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │  Model Loading     │  │  Prediction Logic  │            │
│  └────────────────────┘  └────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ML Models & Preprocessing                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Voting   │  │ Pipeline │  │ K-Means  │  │ XGBoost  │   │
│  │Classifier│  │   (RF)   │  │Clustering│  │Regressor │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  • IBM HR Analytics Dataset                                  │
│  • Stack Overflow Developer Survey                           │
│  • Employee Promotion Dataset                                │
│  • Career Mapping Dataset                                    │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Components

#### **Frontend Layer (Streamlit)**
- Main dashboard (`app.py`) with navigation
- Six dedicated model pages
- Interactive widgets and forms
- Plotly visualizations
- CSV upload/download functionality

#### **Inference Layer (Python Modules)**
- Six inference modules (`src/inference/`)
- Input validation and preprocessing
- Model loading and caching
- Error handling and logging
- Schema documentation

#### **ML Layer (Trained Models)**
- Serialized models (`.pkl` files)
- Preprocessing pipelines
- StandardScalers and encoders
- Model metadata and versioning

---

## 🤖 Models

### 1. 👥 Employee Attrition Prediction

**Type:** Binary Classification (Voting Classifier)  
**Purpose:** Predict if an employee is likely to leave the organization

**Features (13):**
- Age, Distance from Home, Monthly Income
- Environment, Job, Relationship Satisfaction (1-4 scale)
- Job Involvement, Work-Life Balance (1-4 scale)
- Years at Company, in Current Role, Since Promotion, With Manager
- Overtime (Yes/No)

**Output:** 
- Prediction: "Attrition" or "No Attrition"
- Attrition probability (0-100%)
- Risk level indicator

**Model:** Voting Classifier (Ensemble)  
**File:** `best_attrition_prediction_model_voting.pkl`

---

### 2. 📈 Promotion Eligibility Prediction

**Type:** Binary Classification (Random Forest Pipeline)  
**Purpose:** Assess if an employee is ready for promotion

**Features (10):**
- **Numeric (6):** Age, Length of Service, Previous Year Rating, No. of Trainings, Avg Training Score, Awards Won
- **Categorical (4):** Gender, Department, Education, Recruitment Channel

**Output:**
- Prediction: "Promoted" or "Not Promoted"
- Promotion probability
- Eligibility score

**Model:** Random Forest with preprocessing pipeline  
**File:** `hr_promotion_model_full_pipeline.pkl`

---

### 3. 🎯 Skill Composition Clustering

**Type:** Unsupervised Clustering (K-Means)  
**Purpose:** Segment employees into skill-based groups

**Features (17 Technical Skills):**
- Database Fundamentals, Computer Architecture
- Distributed Computing, Cyber Security, Networking
- Software Development, Programming, Project Management
- Computer Forensics, Technical Communication
- AI/ML, Software Engineering, Business Analysis
- Communication Skills, Data Science, Troubleshooting, Graphics Design

**Output:**
- Cluster assignment (0-9)
- Cluster name and description
- Top 3 skills for the cluster
- Distance to cluster center

**Clusters (10):**
0. AI-Driven Business & Systems Analysis
1. Communication-Focused Creative Support
2. Computational Architecture & Data Science
3. Technical Support & Database Operations
4. Technical Communication & Digital Forensics
5. Software Engineering Core Competency
6. Programming-Focused Technical Track
7. Network Engineering Competency
8. Project & Management Leadership
9. Cybersecurity Engineering

**Model:** K-Means (10 clusters)  
**File:** `skill_composition_kmeans_model.pkl`

---

### 4. ⭐ Performance Classification

**Type:** Multi-class Classification (Gradient Boosting)  
**Purpose:** Predict employee performance rating

**Features (7):**
- Job Satisfaction (1-4)
- Environment Satisfaction (1-4)
- Work-Life Balance (1-4)
- Monthly Income
- Years at Company
- Education Level (1-5)
- Training Times Last Year

**Output:**
- Performance Rating: "Below Average" (2), "Average" (3), or "Above Average" (4)
- Class probabilities for each rating
- Performance score

**Model:** Gradient Boosting Classifier  
**File:** `gradient_boosting_performance_classification_model.pkl`  
**Scaler:** `Scalers/scaler.pkl`

---

### 5. 💰 Salary Prediction

**Type:** Regression (XGBoost)  
**Purpose:** Estimate annual developer salary in USD

**Features (6):**
- Years of Professional Coding Experience
- Country
- Education Level
- Employment Type
- Developer Type (Primary Role)
- Primary Programming Language

**Output:**
- Predicted Annual Salary (USD)
- Salary breakdown (yearly, monthly, weekly, hourly)
- Confidence interval

**Model:** XGBoost Regressor  
**File:** `best_salary_model_XGBRegressor.pkl`

**Data Source:** Stack Overflow Developer Survey 2023

---

### 6. 🎯 Employee Engagement Clustering

**Type:** Unsupervised Clustering (K-Means)  
**Purpose:** Segment employees by engagement level

**Features (10):**
- Age, Monthly Income, Job Level
- Years at Company, Total Working Years
- Promotion Rate (calculated)
- Department (encoded), Job Role (encoded)
- Job Involvement, Stock Option Level

**Output:**
- Cluster assignment (0-3)
- Engagement level name
- Cluster characteristics
- Recommendations

**Clusters (4):**
0. **Highly Engaged & Loyal** - Senior employees with high satisfaction and loyalty
1. **Actively Disengaged** - At-risk employees needing intervention
2. **Moderately Engaged** - Stable performers with growth potential
3. **New & Enthusiastic** - Early-career employees with high potential

**Model:** K-Means (4 clusters)  
**Files:** `employee_engagement_clustering.pkl`, `employee_engagement_scaler.pkl`

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/kareemtaha3/AI-Powered-HR-Analytics-Dashboard.git
cd AI-Powered-HR-Analytics-Dashboard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly xgboost openpyxl
```

**Core Dependencies:**
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `joblib>=1.3.0` - Model serialization
- `plotly>=5.17.0` - Interactive visualizations
- `xgboost>=2.0.0` - Gradient boosting
- `openpyxl>=3.1.0` - Excel file support

### Step 4: Verify Installation

```bash
# Check if all models exist
ls -l Models/

# Expected files:
# - best_attrition_prediction_model_voting.pkl
# - hr_promotion_model_full_pipeline.pkl
# - skill_composition_kmeans_model.pkl
# - gradient_boosting_performance_classification_model.pkl
# - best_salary_model_XGBRegressor.pkl
# - employee_engagement_clustering.pkl
```

---

## 🚀 Usage

### Running the Dashboard

```bash
cd src
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Dashboard

#### **Main Dashboard**
- Overview of all 6 models
- Quick navigation buttons
- Model statistics and metrics

#### **Single Prediction**
1. Select a model from the sidebar
2. Fill in the input fields using sliders/dropdowns
3. Click "Predict" button
4. View results with visualizations

#### **Batch Prediction**
1. Navigate to "Batch Prediction" tab
2. Download the CSV template
3. Fill in employee data
4. Upload the CSV file
5. View aggregated results and download predictions

#### **Model Information**
- View input schema and required columns
- See feature descriptions
- Access example inputs
- Understand model outputs

### Example: Single Prediction

```python
# Attrition Prediction Example
Input:
  - Age: 35
  - Monthly Income: $5,000
  - Job Satisfaction: 3 (out of 4)
  - Overtime: No
  - Years at Company: 5

Output:
  - Prediction: "No Attrition"
  - Attrition Risk: 15%
  - Risk Level: Low
```

### Example: Batch Prediction

```csv
Age,MonthlyIncome,JobSatisfaction,OverTime,YearsAtCompany,...
35,5000,3,No,5,...
42,8000,4,No,10,...
28,3500,2,Yes,2,...
```

Results downloaded as `predictions_[timestamp].csv` with all input features + predictions.

---

## 📁 Project Structure

```
project/
│
├── README.md                           # This file
├── PROJECT_STRUCTURE.md                # Detailed structure documentation
├── STREAMLIT_PAGES_SUMMARY.md          # Pages implementation summary
├── SETUP_CHECKLIST.md                  # Setup verification guide
├── .gitignore                          # Git ignore rules
│
├── Data/                               # Datasets (gitignored)
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.xls
│   ├── IBM_Engagement_Clustering_Ready.csv
│   ├── hr model2/
│   │   └── HR_Analytics.csv
│   └── archive/
│       ├── survey_results_public.csv
│       └── survey_results_schema.csv
│
├── Models/                             # Trained ML models (6 files)
│   ├── best_attrition_prediction_model_voting.pkl
│   ├── hr_promotion_model_full_pipeline.pkl
│   ├── skill_composition_kmeans_model.pkl
│   ├── gradient_boosting_performance_classification_model.pkl
│   ├── best_salary_model_XGBRegressor.pkl
│   └── employee_engagement_clustering.pkl
│
├── Scalers/                            # Preprocessing scalers
│   └── scaler.pkl
│
├── notebooks/                          # Jupyter notebooks (EDA & training)
│   ├── employee_attrition_prediction_model.ipynb
│   ├── promotion_eligibility_prediction.ipynb
│   ├── employee_career_clustring.ipynb
│   ├── employee_career_clustring_EDA.ipynb
│   ├── employee_performance_cassification.ipynb
│   ├── Model_4_Salary_Prediction_with_EDA.ipynb
│   └── Employee_Engagement_Clustering.ipynb
│
└── src/                                # Source code
    ├── __init__.py
    ├── app.py                          # Main Streamlit application
    │
    ├── inference/                      # Inference modules (6 models)
    │   ├── __init__.py
    │   ├── attrition_prediction_model.py
    │   ├── promotion_eligibility_model.py
    │   ├── skill_clustring_model.py
    │   ├── performance_classification_model.py
    │   ├── salary_prediction_model.py
    │   └── employee_engagement_model.py
    │
    └── pages/                          # Streamlit pages (6 pages)
        ├── 01_attrition_prediction.py
        ├── 02_promotion_eligibility.py
        ├── 03_career_clustering.py
        ├── 04_performance_classification.py
        ├── 05_salary_prediction.py
        └── 06_employee_engagement.py
```

### Key Directories

- **`Data/`**: Contains raw and processed datasets (excluded from git)
- **`Models/`**: Serialized trained models and pipelines
- **`Scalers/`**: Preprocessing transformers (StandardScaler, etc.)
- **`notebooks/`**: Jupyter notebooks for EDA, training, and evaluation
- **`src/`**: Production-ready code
  - **`inference/`**: Model loading and prediction logic
  - **`pages/`**: Streamlit UI pages

---

## 📊 Dataset Information

### 1. IBM HR Analytics Dataset
- **Source:** IBM Watson Analytics
- **Records:** ~1,470 employees
- **Features:** 35 attributes
- **Used for:** Attrition, Performance, Engagement models
- **File:** `WA_Fn-UseC_-HR-Employee-Attrition.xls`

### 2. Stack Overflow Developer Survey 2023
- **Source:** Stack Overflow Annual Survey
- **Records:** ~90,000 developers
- **Features:** Salary, skills, demographics
- **Used for:** Salary Prediction model
- **File:** `archive/survey_results_public.csv`

### 3. Employee Promotion Dataset
- **Source:** Kaggle
- **Records:** ~50,000 employees
- **Features:** Performance, training, awards
- **Used for:** Promotion Eligibility model
- **File:** `hr model2/HR_Analytics.csv`
- **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/muhammadimran112233/employees-evaluation-for-promotion)

### 4. Career Mapping Dataset
- **Records:** Technical skill ratings
- **Features:** 17 skill competencies
- **Used for:** Skill Clustering model
- **File:** `CareerMap-*.csv`

---

## 🔧 Technical Details

### Machine Learning Pipeline

#### **1. Data Preprocessing**
```python
# Example from Attrition model
1. Handle missing values
2. Encode categorical features (LabelEncoder/OneHotEncoder)
3. Scale numerical features (StandardScaler)
4. Feature engineering (ratios, interactions)
5. Outlier detection (IsolationForest)
```

#### **2. Model Training**
```python
# Classification models
- Train/test split (80/20)
- Cross-validation (5-fold)
- Hyperparameter tuning (GridSearchCV)
- Model selection (best F1/accuracy)

# Clustering models
- Elbow method for K selection
- Silhouette score analysis
- Feature normalization
- Cluster profiling
```

#### **3. Model Evaluation**
```python
# Classification metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, Confusion Matrix
- Classification Report

# Regression metrics
- RMSE, MAE, R² score
- Residual analysis

# Clustering metrics
- Silhouette score
- Inertia
- Cluster separation
```

### Model Serialization

All models are saved using `joblib`:
```python
import joblib

# Save model
joblib.dump(model, 'Models/model_name.pkl')

# Load model
model = joblib.load('Models/model_name.pkl')
```

### Inference API

Each inference module provides:
```python
# Standard interface
predict_*()           # Main prediction function
validate_input()      # Input validation
get_input_schema()    # Schema documentation
example_input()       # Example inputs
load_model()          # Model loading
```

### Performance Optimization

- **Model Caching:** Streamlit's `@st.cache_resource` for model loading
- **Lazy Loading:** Models loaded only when needed
- **Batch Processing:** Vectorized predictions for CSV uploads
- **Error Handling:** Comprehensive validation and error messages

---

## 🛠️ Development

### Adding a New Model

1. **Train the model** in a Jupyter notebook
2. **Save the model** to `Models/` directory
3. **Create inference module** in `src/inference/`
4. **Create Streamlit page** in `src/pages/`
5. **Update `app.py`** navigation
6. **Update `__init__.py`** exports
7. **Test thoroughly**

### Code Quality

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write comprehensive error messages
- Add input validation

### Testing

```bash
# Test individual models
python -c "from src.inference.attrition_prediction_model import *; print(predict_attrition(example_input()))"

# Test Streamlit pages
streamlit run src/pages/01_attrition_prediction.py
```

---

## 📈 Model Performance

| Model | Type | Metric | Score |
|-------|------|--------|-------|
| Attrition | Classification | Accuracy | ~87% |
| Promotion | Classification | F1-Score | ~85% |
| Skill Clustering | Clustering | Silhouette | ~0.45 |
| Performance | Classification | Accuracy | ~82% |
| Salary | Regression | R² | ~0.76 |
| Engagement | Clustering | Silhouette | ~0.38 |

*Note: Actual performance may vary based on dataset and training parameters*

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas

- Add new HR analytics models
- Improve model accuracy
- Enhance UI/UX
- Add more visualizations
- Write unit tests
- Improve documentation
- Add API endpoints

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Kareem Taha** - [GitHub](https://github.com/kareemtaha3)

---

## 🙏 Acknowledgments

- **IBM Watson Analytics** for the HR dataset
- **Stack Overflow** for the developer survey data
- **Kaggle** community for datasets and inspiration
- **Streamlit** team for the amazing framework
- **scikit-learn** developers for ML tools

---

## 📞 Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/kareemtaha3/AI-Powered-HR-Analytics-Dashboard/issues)
- **Email:** Contact through GitHub profile
- **Documentation:** See `PROJECT_STRUCTURE.md` and `STREAMLIT_PAGES_SUMMARY.md`

---

## 🔮 Future Enhancements

- [ ] Add REST API endpoints
- [ ] Implement model retraining pipeline
- [ ] Add A/B testing framework
- [ ] Create Docker container
- [ ] Add user authentication
- [ ] Implement model monitoring
- [ ] Add explainability (SHAP values)
- [ ] Create mobile-responsive UI
- [ ] Add data drift detection
- [ ] Implement automated testing

---

## 📚 Additional Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure
- **[STREAMLIT_PAGES_SUMMARY.md](STREAMLIT_PAGES_SUMMARY.md)** - Page implementation details
- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Setup verification guide

---

<div align="center">

**⭐ Star this repository if you find helpful!**

Made with ❤️ and ☕ by Kareem Taha

</div>
