"""
Salary Prediction Page
Predicts developer annual salary using ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.salary_prediction_model import (
    predict_salary,
    example_input,
    REQUIRED_COLUMNS,
    COMMON_COUNTRIES,
    COMMON_EDUCATION_LEVELS,
    COMMON_EMPLOYMENT_TYPES,
    COMMON_DEV_TYPES,
    COMMON_LANGUAGES,
    get_input_schema
)

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Developer Salary Prediction")
st.markdown("""
Estimate annual developer salary based on experience, location, education, 
role, and technical skills using Stack Overflow survey data.
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Developer Salary Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Developer Profile")
        
        # Experience
        st.write("**Experience**")
        years_code_pro = st.slider(
            "Years of Professional Coding",
            min_value=0, max_value=50, value=3, step=1
        )
        
        # Location
        st.write("**Location**")
        country = st.selectbox(
            "Country",
            COMMON_COUNTRIES,
            index=0
        )
        
        # Education
        st.write("**Education**")
        ed_level = st.selectbox(
            "Education Level",
            COMMON_EDUCATION_LEVELS,
            index=0
        )
        
        # Employment
        st.write("**Employment**")
        employment = st.selectbox(
            "Employment Status",
            COMMON_EMPLOYMENT_TYPES,
            index=0
        )
        
        # Role & Skills
        st.write("**Role & Technical Skills**")
        dev_type = st.selectbox(
            "Developer Type/Role",
            COMMON_DEV_TYPES,
            index=0
        )
        
        lang_first = st.selectbox(
            "Primary Programming Language",
            COMMON_LANGUAGES,
            index=0
        )
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Prepare input data
        input_data = {
            'YearsCodePro': years_code_pro,
            'Country': country,
            'EdLevel': ed_level,
            'Employment': employment,
            'DevType_first': dev_type,
            'Lang_first': lang_first
        }
        
        if st.button("💰 Predict Salary", use_container_width=True, type="primary"):
            try:
                result = predict_salary(input_data)
                
                # Display prediction
                predicted_salary = result['predicted_salary']
                currency = result['currency']
                
                # Format salary
                salary_formatted = f"${predicted_salary:,.0f}"
                
                st.success(f"✅ **Predicted Annual Salary**", icon="💰")
                st.metric("Estimated Salary", salary_formatted, delta=None)
                
                # Salary gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_salary,
                    title={'text': "Annual Salary (USD)"},
                    gauge={
                        'axis': {'range': [0, min(predicted_salary * 2, 500000)]},
                        'bar': {'color': "#51cf66"},
                        'steps': [
                            {'range': [0, 50000], 'color': "#2d2d2d"},
                            {'range': [50000, 100000], 'color': "#1a4d6d"},
                            {'range': [100000, 150000], 'color': "#2d5016"},
                            {'range': [150000, 500000], 'color': "#5c4813"}
                        ]
                    }
                ))
                fig.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Salary breakdown
                st.subheader("📊 Salary Breakdown")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    monthly = predicted_salary / 12
                    st.metric("Monthly", f"${monthly:,.0f}")
                
                with col_b:
                    weekly = predicted_salary / 52
                    st.metric("Weekly", f"${weekly:,.0f}")
                
                with col_c:
                    hourly = predicted_salary / (52 * 40)
                    st.metric("Hourly", f"${hourly:,.0f}")
                
                # Show input summary
                st.subheader("📋 Developer Profile Summary")
                summary_df = pd.DataFrame({
                    'Factor': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.session_state.last_prediction = result
                st.session_state.last_input = input_data
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Display salary factors
        st.subheader("⚡ Salary Factors")
        with st.expander("View key factors", expanded=False):
            st.markdown("""
            **Key factors affecting salary:**
            
            - **Experience**: More years of professional coding = higher salary
            - **Location**: Country/region significantly impacts compensation
            - **Education**: Higher education often correlates with higher pay
            - **Employment Type**: Full-time employment typically pays more
            - **Developer Role**: Specialized roles (ML, DevOps) often command premium
            - **Programming Language**: Some languages (Go, Rust) associated with higher salaries
            """)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Salary Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple developer profiles to predict salaries for all at once.
    
    **Required columns:**
    - YearsCodePro (numeric)
    - Country (string)
    - EdLevel (string)
    - Employment (string)
    - DevType_first (string)
    - Lang_first (string)
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.subheader("📄 Uploaded Data")
            st.dataframe(batch_df.head(10), use_container_width=True)
            st.info(f"Total records: {len(batch_df)}")
            
            if st.button("🚀 Predict All Salaries", use_container_width=True, type="primary"):
                try:
                    from inference.salary_prediction_model import predict_salary
                    
                    # Make batch predictions
                    results = []
                    for idx, row in batch_df.iterrows():
                        try:
                            result = predict_salary(row.to_dict())
                            results.append({
                                'Row': idx + 1,
                                'Predicted_Salary': result['predicted_salary'],
                                'Currency': result['currency']
                            })
                        except Exception as e:
                            results.append({
                                'Row': idx + 1,
                                'Predicted_Salary': None,
                                'Currency': f"Error: {str(e)}"
                            })
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    valid_results = results_df[results_df['Predicted_Salary'].notna()]
                    
                    if len(valid_results) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_salary = valid_results['Predicted_Salary'].mean()
                            st.metric("Average Salary", f"${avg_salary:,.0f}")
                        
                        with col2:
                            median_salary = valid_results['Predicted_Salary'].median()
                            st.metric("Median Salary", f"${median_salary:,.0f}")
                        
                        with col3:
                            min_salary = valid_results['Predicted_Salary'].min()
                            st.metric("Min Salary", f"${min_salary:,.0f}")
                        
                        with col4:
                            max_salary = valid_results['Predicted_Salary'].max()
                            st.metric("Max Salary", f"${max_salary:,.0f}")
                        
                        # Distribution chart
                        fig = px.histogram(
                            valid_results,
                            x='Predicted_Salary',
                            nbins=30,
                            title='Salary Distribution',
                            labels={'Predicted_Salary': 'Annual Salary (USD)'},
                            template="plotly_dark",
                            color_discrete_sequence=['#4dabf7']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="salary_predictions.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
                    
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")

# Tab 3: Model Info
with tab3:
    st.header("📚 Model Information")
    
    st.subheader("Model Details")
    st.markdown("""
    **Model Type:** XGBoost Regressor (Best performing model)
    
    **Purpose:** Predict annual developer salary in USD
    
    **Data Source:** Stack Overflow Developer Survey 2023
    
    **Features Used:**
    - Years of professional coding experience
    - Country/location
    - Education level
    - Employment type
    - Developer type/role
    - Primary programming language
    """)
    
    st.subheader("Input Schema")
    schema = get_input_schema()
    schema_df = pd.DataFrame([
        {
            "Column": col,
            "Type": schema['feature_info'][col]['type'],
            "Description": schema['feature_info'][col]['description']
        }
        for col in REQUIRED_COLUMNS
    ])
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    
    st.subheader("Example Input")
    example = example_input()
    example_df = pd.DataFrame([example])
    st.dataframe(example_df, use_container_width=True)
    
    st.subheader("Model Performance")
    st.markdown("""
    The model has been trained and evaluated using:
    - **Algorithm**: XGBoost Regressor
    - **Preprocessing**: OneHotEncoder for categorical features, StandardScaler for numeric
    - **Evaluation Metrics**: MAE, RMSE, R² score
    - **Training Data**: Stack Overflow Developer Survey 2023
    
    The model provides salary estimates with consideration for:
    - Geographic location (country-specific compensation levels)
    - Experience level and seniority
    - Educational background
    - Technical specialization
    - Programming language expertise
    """)
    
    st.subheader("Use Cases")
    st.markdown("""
    - **Compensation Planning**: Benchmark salaries for different roles
    - **Hiring Strategy**: Estimate competitive salary offers
    - **Career Planning**: Understand salary potential for different paths
    - **Budget Forecasting**: Estimate personnel costs
    - **Market Analysis**: Compare compensation across regions and roles
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    - Predictions are based on survey data and may not reflect all markets
    - Currency is USD; conversion needed for local currency
    - Does not account for benefits, equity, or bonuses
    - Regional cost of living differences within countries not captured
    - Model trained on 2023 data; may need updates for current market
    """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Salary Prediction Model**
    
    Estimates developer annual salary based on:
    - Professional experience
    - Location (country)
    - Education level
    - Employment type
    - Developer role
    - Programming language
    
    Use this tool to:
    - Plan compensation packages
    - Benchmark market rates
    - Evaluate career opportunities
    - Forecast hiring budgets
    """)
    
    st.subheader("Quick Tips")
    st.markdown("""
    💡 **Tips for accurate predictions:**
    - Select your primary role and language
    - Be honest about experience level
    - Choose your actual country
    - Consider full-time employment baseline
    """)
    
    if st.button("Show Example"):
        st.session_state.show_example = True
        example = example_input()
        st.json(example)
