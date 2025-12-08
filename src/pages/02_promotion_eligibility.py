"""
Promotion Eligibility Prediction Page
Predicts employee promotion eligibility using ML model
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

from inference.promotion_eligibility_model import (
    predict_promotion,
    example_input,
    REQUIRED_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    get_input_schema
)

st.set_page_config(
    page_title="Promotion Eligibility",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Promotion Eligibility Prediction")
st.markdown("""
Predict employee promotion eligibility based on performance metrics, tenure, and career development.
This model helps identify high-potential employees ready for advancement.
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Employee Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Data")
        print(CATEGORICAL_FEATURES, CATEGORY_VALUES)
        # Numeric inputs
        st.write("**Basic Information**")
        age = st.slider("Age", min_value=18, max_value=65, value=35, step=1)
        gender = st.selectbox("Gender", [g.capitalize() for g in CATEGORY_VALUES['gender']])
        department = st.selectbox("Department", [d.capitalize() for d in CATEGORY_VALUES['department']])
        education = st.selectbox("Education", [e.capitalize() for e in CATEGORY_VALUES['education']])
        recruitment_channel = st.selectbox("Recruitment Channel", [r.capitalize() for r in CATEGORY_VALUES['recruitment_channel']])
        
        st.write("**Experience & Performance**")
        length_of_service = st.slider("Length of Service (years)", min_value=0, max_value=40, value=5, step=1)
        previous_year_rating = st.slider("Previous Year Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        
        st.write("**Training & Development**")
        no_of_trainings = st.slider("Number of Trainings", min_value=0, max_value=10, value=3, step=1)
        avg_training_score = st.slider("Average Training Score (0-100)", min_value=0, max_value=100, value=75, step=5)
        
        st.write("**Recognition**")
        awards_won = st.slider("Awards Won", min_value=0, max_value=5, value=1, step=1)
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Prepare input data - convert categorical values to lowercase
        input_data = {
            'age': age,
            'gender': gender.lower(),
            'department': department.lower(),
            'education': education.lower(),
            'recruitment_channel': recruitment_channel.lower(),
            'length_of_service': length_of_service,
            'previous_year_rating': previous_year_rating,
            'no_of_trainings': no_of_trainings,
            'avg_training_score': avg_training_score,
            'awards_won': awards_won
        }
        
        if st.button("🚀 Predict Promotion Eligibility", use_container_width=True, type="primary"):
            try:
                result = predict_promotion(input_data)
                
                # Display prediction
                prediction = result['prediction']
                probability = result.get('promotion_probability', None)
                
                if prediction == "Promoted":
                    st.success(f"✅ **ELIGIBLE FOR PROMOTION**", icon="✅")
                    prob_color = "green"
                    prob_display = "High"
                else:
                    st.info(f"⏳ **NOT YET ELIGIBLE**", icon="ℹ️")
                    prob_color = "orange"
                    prob_display = "Low"
                
                if probability is not None:
                    # Display probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        title={'text': "Promotion Eligibility Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': prob_color},
                            'steps': [
                                {'range': [0, 40], 'color': "#5c1a1a"},
                                {'range': [40, 70], 'color': "#5c4813"},
                                {'range': [70, 100], 'color': "#2d5016"}
                            ],
                            'threshold': {
                                'line': {'color': "#4dabf7", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    fig.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Eligibility Score", f"{probability*100:.1f}%")
                
                # Show input summary
                st.subheader("📊 Employee Profile Summary")
                summary_df = pd.DataFrame({
                    'Factor': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.session_state.last_prediction = result
                st.session_state.last_input = input_data
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Display promotion factors
        st.subheader("⚡ Promotion Factors")
        with st.expander("View key factors", expanded=False):
            st.info("""
            **Positive Promotion Indicators:**
            - High previous year rating (3+)
            - Consistent training participation (3+ trainings)
            - Good average training scores (70+)
            - Awards and recognition
            - Adequate tenure in service
            
            **Considerations:**
            - Department and role fit
            - Educational background
            - Career progression pathway
            """)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Promotion Eligibility Assessment")
    st.write("Upload a CSV file to assess promotion eligibility for multiple employees.")
    
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
                if st.button("🚀 Assess All Employees", type="primary", use_container_width=True):
                    try:
                        predictions = []
                        probabilities = []
                        
                        with st.spinner("Processing predictions..."):
                            for idx, row in df.iterrows():
                                input_dict = row[REQUIRED_COLUMNS].to_dict()
                                result = predict_promotion(input_dict)
                                predictions.append(result['prediction'])
                                if result.get('promotion_probability'):
                                    probabilities.append(result['promotion_probability'])
                                else:
                                    probabilities.append(None)
                        
                        # Add results to dataframe
                        results_df = df.copy()
                        results_df['Promotion_Eligible'] = predictions
                        results_df['Eligibility_Score'] = probabilities
                        
                        st.success(f"✅ Assessment completed for {len(df)} employees")
                        
                        # Display results
                        st.subheader("Assessment Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        eligible_count = (results_df['Promotion_Eligible'] == 'Promoted').sum()
                        with col1:
                            st.metric("Promotion Eligible", eligible_count, f"{eligible_count/len(df)*100:.1f}%")
                        with col2:
                            st.metric("Not Yet Eligible", len(df) - eligible_count, f"{(len(df)-eligible_count)/len(df)*100:.1f}%")
                        with col3:
                            avg_score = np.nanmean([p for p in probabilities if p is not None])
                            st.metric("Avg Eligibility Score", f"{avg_score*100:.1f}%", "Across all employees")
                        
                        # Visualization
                        st.subheader("📊 Results Visualization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = px.pie(
                                values=[eligible_count, len(df) - eligible_count],
                                names=['Promotion Eligible', 'Not Yet Eligible'],
                                title="Promotion Eligibility Distribution",
                                color_discrete_map={'Promotion Eligible': '#51cf66', 'Not Yet Eligible': '#ff8787'},
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Histogram of scores
                            if any(p is not None for p in probabilities):
                                valid_probs = [p for p in probabilities if p is not None]
                                fig = go.Figure(data=[
                                    go.Histogram(x=valid_probs, nbinsx=20, name='Eligibility Score', marker_color='#4dabf7')
                                ])
                                fig.update_layout(
                                    title="Distribution of Eligibility Scores",
                                    xaxis_title="Score",
                                    yaxis_title="Count",
                                    showlegend=False,
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Assessment Results (CSV)",
                            data=csv,
                            file_name="promotion_eligibility_assessment.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Batch assessment failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")

# Tab 3: Model Info
with tab3:
    st.header("Model Information")
    
    st.subheader("📊 About This Model")
    st.info("""
    The Promotion Eligibility Prediction model is trained on historical HR data to identify employees
    who are ready for promotion based on their performance, experience, training, and development.
    
    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Training Data: Employee promotion history
    - Features: 10 features (6 numeric, 4 categorical)
    - Output: Binary classification (Promoted/Not Promoted) with probability score
    """)
    
    st.subheader("📋 Required Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Features**")
        for feature in NUMERIC_FEATURES:
            st.write(f"- {feature}")
    
    with col2:
        st.write("**Categorical Features**")
        for feature in CATEGORICAL_FEATURES:
            allowed_values = CATEGORY_VALUES.get(feature, [])
            st.write(f"- {feature}: {', '.join(allowed_values)}")
    
    st.subheader("📊 Example Input")
    example = example_input()
    st.json(example)
    
    st.subheader("🎯 Output Interpretation")
    st.write("""
    - **Score 0-40%**: Not eligible for promotion - needs more development
    - **Score 40-70%**: Approaching eligibility - consider development plans
    - **Score 70%+**: Eligible for promotion - ready for advancement
    """)
    
    st.subheader("📌 Feature Descriptions")
    feature_descriptions = {
        'age': 'Employee age in years',
        'gender': 'Gender (m=Male, f=Female)',
        'department': 'Department assignment',
        'education': 'Education level (Bachelor, Master, PHD)',
        'recruitment_channel': 'How employee was recruited (sourcing, referred, campus)',
        'length_of_service': 'Total years employed at company',
        'previous_year_rating': 'Performance rating from previous year (1-5)',
        'no_of_trainings': 'Number of trainings completed',
        'avg_training_score': 'Average score in training programs (0-100)',
        'awards_won': 'Number of awards and recognitions'
    }
    
    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}**: {description}")
    
    st.subheader("✅ Promotion Readiness Checklist")
    checklist = {
        "Performance Rating": "Should be 3 or above",
        "Training Participation": "Typically 3+ trainings completed",
        "Training Scores": "Average 70+ on training assessments",
        "Recognition": "At least 1 award or recognition",
        "Tenure": "Adequate company experience (2+ years)"
    }
    
    for criterion, description in checklist.items():
        st.write(f"- ✓ {criterion}: {description}")
