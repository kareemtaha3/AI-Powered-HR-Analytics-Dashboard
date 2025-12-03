"""
Employee Attrition Prediction Page
Predicts likelihood of employee attrition using ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.attrition_prediction_model import (
    predict_attrition,
    example_input,
    REQUIRED_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    get_input_schema
)

st.set_page_config(
    page_title="Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Employee Attrition Prediction")
st.markdown("""
Predict the likelihood of employee attrition. This model analyzes various employee factors 
to identify at-risk employees who may leave the organization.
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Input Schema"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Employee Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Data")
        
        # Numeric inputs
        age = st.slider("Age", min_value=18, max_value=65, value=35, step=1)
        distance_from_home = st.slider("Distance From Home (km)", min_value=0, max_value=30, value=10, step=1)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
        years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=5, step=1)
        years_in_current_role = st.slider("Years In Current Role", min_value=0, max_value=20, value=2, step=1)
        years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=20, value=1, step=1)
        years_with_curr_manager = st.slider("Years With Current Manager", min_value=0, max_value=20, value=3, step=1)
        
        st.subheader("🎯 Satisfaction Metrics (1-4 scale)")
        col_a, col_b = st.columns(2)
        with col_a:
            environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=3)
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=3)
            job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=3)
        with col_b:
            relationship_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4, value=3)
            work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=4, value=3)
        
        st.subheader("📅 Work Schedule")
        over_time = st.selectbox("Overtime", options=["Yes", "No"], index=1)
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Prepare input data
        input_data = {
            'Age': age,
            'DistanceFromHome': distance_from_home,
            'MonthlyIncome': monthly_income,
            'EnvironmentSatisfaction': environment_satisfaction,
            'OverTime': over_time,
            'JobInvolvement': job_involvement,
            'RelationshipSatisfaction': relationship_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'JobSatisfaction': job_satisfaction
        }
        
        if st.button("🚀 Predict Attrition", use_container_width=True, type="primary"):
            try:
                result = predict_attrition(input_data)
                
                # Display prediction
                prediction = result['prediction']
                probability = result.get('attrition_probability', None)
                
                if prediction == "Attrition":
                    st.error(f"⚠️ **HIGH RISK**: Employee likely to leave", icon="⚠️")
                    risk_color = "red"
                    risk_level = "HIGH"
                else:
                    st.success(f"✅ **LOW RISK**: Employee likely to stay", icon="✅")
                    risk_color = "green"
                    risk_level = "LOW"
                
                if probability is not None:
                    # Display probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        title={'text': "Attrition Risk Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': risk_color},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgreen"},
                                {'range': [33, 66], 'color': "lightyellow"},
                                {'range': [66, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Attrition Probability", f"{probability*100:.1f}%", f"Risk Level: {risk_level}")
                
                # Show input summary
                st.subheader("📊 Employee Profile Summary")
                summary_df = pd.DataFrame({
                    'Factor': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Store result in session state
                st.session_state.last_prediction = result
                st.session_state.last_input = input_data
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Display risk factors
        st.subheader("⚡ Key Risk Factors")
        with st.expander("View contributing factors", expanded=False):
            st.info("""
            **Potential Attrition Risk Factors:**
            - Low job satisfaction or environment satisfaction
            - Recent or frequent promotion delays
            - High overtime workload
            - Long distance from home
            - Lower work-life balance scores
            - Short tenure at company or in role
            """)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with employee data to get predictions for multiple employees.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} employees")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check for required columns
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                if st.button("🚀 Predict for All Employees", type="primary", use_container_width=True):
                    try:
                        predictions = []
                        probabilities = []
                        
                        with st.spinner("Processing predictions..."):
                            for idx, row in df.iterrows():
                                input_dict = row[REQUIRED_COLUMNS].to_dict()
                                result = predict_attrition(input_dict)
                                predictions.append(result['prediction'])
                                if result.get('attrition_probability'):
                                    probabilities.append(result['attrition_probability'])
                                else:
                                    probabilities.append(None)
                        
                        # Add results to dataframe
                        results_df = df.copy()
                        results_df['Attrition_Prediction'] = predictions
                        results_df['Attrition_Probability'] = probabilities
                        
                        st.success(f"✅ Predictions completed for {len(df)} employees")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        attrition_count = (results_df['Attrition_Prediction'] == 'Attrition').sum()
                        with col1:
                            st.metric("At-Risk Employees", attrition_count, f"{attrition_count/len(df)*100:.1f}%")
                        with col2:
                            st.metric("Safe Employees", len(df) - attrition_count, f"{(len(df)-attrition_count)/len(df)*100:.1f}%")
                        with col3:
                            avg_prob = np.mean(probabilities)
                            st.metric("Avg Risk Score", f"{avg_prob*100:.1f}%", "Across all employees")
                        
                        # Visualization
                        st.subheader("📊 Results Visualization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = px.pie(
                                values=[attrition_count, len(df) - attrition_count],
                                names=['At-Risk (Likely Attrition)', 'Safe (Likely to Stay)'],
                                title="Employee Distribution",
                                color_discrete_map={'At-Risk (Likely Attrition)': '#EF553B', 'Safe (Likely to Stay)': '#00CC96'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Histogram of probabilities
                            fig = go.Figure(data=[
                                go.Histogram(x=probabilities, nbinsx=20, name='Attrition Probability')
                            ])
                            fig.update_layout(
                                title="Distribution of Attrition Probabilities",
                                xaxis_title="Probability",
                                yaxis_title="Count",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name="attrition_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")

# Tab 3: Input Schema
with tab3:
    st.header("Input Schema & Documentation")
    
    schema = get_input_schema()
    
    st.subheader("📋 Required Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Features**")
        for feature in schema['numeric_features']:
            st.write(f"- {feature}")
    
    with col2:
        st.write("**Categorical Features**")
        for feature in schema['categorical_features']:
            allowed_values = schema['category_values'].get(feature, [])
            st.write(f"- {feature}: {', '.join(allowed_values)}")
    
    st.subheader("📊 Example Input")
    example = example_input()
    st.json(example)
    
    st.subheader("📌 Feature Descriptions")
    feature_descriptions = {
        'Age': 'Employee age in years',
        'DistanceFromHome': 'Distance from employee home to workplace in km',
        'MonthlyIncome': 'Employee monthly salary in USD',
        'EnvironmentSatisfaction': 'Satisfaction with work environment (1-4 scale)',
        'OverTime': 'Whether employee works overtime (Yes/No)',
        'JobInvolvement': 'Job involvement level (1-4 scale)',
        'RelationshipSatisfaction': 'Satisfaction with relationships at work (1-4 scale)',
        'WorkLifeBalance': 'Work-life balance satisfaction (1-4 scale)',
        'YearsAtCompany': 'Total years employed at company',
        'YearsInCurrentRole': 'Years in current job role',
        'YearsSinceLastPromotion': 'Years since last promotion',
        'YearsWithCurrManager': 'Years working with current manager',
        'JobSatisfaction': 'Overall job satisfaction (1-4 scale)'
    }
    
    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}**: {description}")
