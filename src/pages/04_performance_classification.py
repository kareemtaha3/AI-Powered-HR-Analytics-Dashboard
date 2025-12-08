"""
Performance Classification Page
Predicts employee performance rating using ML model
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

from inference.performance_classification_model import (
    predict_performance,
    example_input,
    REQUIRED_COLUMNS,
    PERFORMANCE_LABELS,
    get_input_schema
)

st.set_page_config(
    page_title="Performance Classification",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ Employee Performance Classification")
st.markdown("""
Predict employee performance rating based on job satisfaction, work environment, 
work-life balance, compensation, tenure, education, and training.
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Employee Performance Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Data")
        
        # Satisfaction & Work-Life Balance
        st.write("**Satisfaction & Work-Life Balance**")
        job_satisfaction = st.slider(
            "Job Satisfaction (1=Low, 4=High)",
            min_value=1, max_value=4, value=3, step=1
        )
        environment_satisfaction = st.slider(
            "Environment Satisfaction (1=Low, 4=High)",
            min_value=1, max_value=4, value=3, step=1
        )
        work_life_balance = st.slider(
            "Work-Life Balance (1=Poor, 4=Excellent)",
            min_value=1, max_value=4, value=3, step=1
        )
        
        # Compensation & Tenure
        st.write("**Compensation & Tenure**")
        monthly_income = st.number_input(
            "Monthly Income (USD)",
            min_value=1000, max_value=20000, value=5000, step=100
        )
        years_at_company = st.slider(
            "Years At Company",
            min_value=0, max_value=40, value=5, step=1
        )
        
        # Education & Training
        st.write("**Education & Training**")
        education = st.slider(
            "Education Level (1=Below College, 5=Doctor)",
            min_value=1, max_value=5, value=3, step=1
        )
        training_times_last_year = st.slider(
            "Training Times Last Year",
            min_value=0, max_value=6, value=2, step=1
        )
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        # Prepare input data
        input_data = {
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'Education': education,
            'TrainingTimesLastYear': training_times_last_year
        }
        
        if st.button("⭐ Predict Performance", use_container_width=True, type="primary"):
            try:
                result = predict_performance(input_data)
                
                # Display prediction
                performance_rating = result['performance_rating']
                performance_label = result['performance_label']
                probabilities = result.get('probabilities', {})
                
                # Color coding
                if performance_rating == 4:
                    st.success(f"✅ **{performance_label}**", icon="⭐")
                    color = "green"
                elif performance_rating == 3:
                    st.info(f"📊 **{performance_label}**", icon="ℹ️")
                    color = "blue"
                else:
                    st.warning(f"⚠️ **{performance_label}**", icon="⚠️")
                    color = "orange"
                
                # Display performance metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Performance Rating", f"{performance_rating}/4")
                with col_b:
                    st.metric("Performance Level", performance_label)
                
                # Display probability distribution if available
                if probabilities:
                    st.subheader("📊 Probability Distribution")
                    
                    # Create bar chart for probabilities
                    prob_df = pd.DataFrame([
                        {"Rating": PERFORMANCE_LABELS.get(k, f"Rating {k}"), "Probability": v}
                        for k, v in probabilities.items()
                    ])
                    
                    fig = px.bar(
                        prob_df,
                        x="Rating",
                        y="Probability",
                        title="Performance Rating Probabilities",
                        color="Probability",
                        color_continuous_scale="Blues",
                        template="plotly_dark"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
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
        
        # Display performance factors
        st.subheader("⚡ Performance Factors")
        with st.expander("View key factors", expanded=False):
            st.markdown("""
            **Key factors affecting performance rating:**
            
            - **Job Satisfaction**: Higher satisfaction correlates with better performance
            - **Environment Satisfaction**: Positive work environment enhances productivity
            - **Work-Life Balance**: Better balance leads to sustained performance
            - **Training**: More training opportunities improve skills and performance
            - **Tenure**: Experience at company affects performance expectations
            - **Income**: Compensation satisfaction impacts motivation
            - **Education**: Educational background provides foundation for performance
            """)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Performance Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple employees to predict performance for all at once.
    
    **Required columns:**
    - JobSatisfaction (1-4)
    - EnvironmentSatisfaction (1-4)
    - WorkLifeBalance (1-4)
    - MonthlyIncome (numeric)
    - YearsAtCompany (numeric)
    - Education (1-5)
    - TrainingTimesLastYear (numeric)
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
            
            if st.button("🚀 Predict All", use_container_width=True, type="primary"):
                try:
                    from inference.performance_classification_model import predict_performance
                    
                    # Make batch predictions
                    results = []
                    for idx, row in batch_df.iterrows():
                        try:
                            result = predict_performance(row.to_dict())
                            results.append({
                                'Row': idx + 1,
                                'Performance_Rating': result['performance_rating'],
                                'Performance_Label': result['performance_label']
                            })
                        except Exception as e:
                            results.append({
                                'Row': idx + 1,
                                'Performance_Rating': None,
                                'Performance_Label': f"Error: {str(e)}"
                            })
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    valid_results = results_df[results_df['Performance_Rating'].notna()]
                    
                    if len(valid_results) > 0:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_rating = valid_results['Performance_Rating'].mean()
                            st.metric("Average Rating", f"{avg_rating:.2f}")
                        
                        with col2:
                            above_avg = len(valid_results[valid_results['Performance_Rating'] == 4])
                            st.metric("Above Average", above_avg)
                        
                        with col3:
                            below_avg = len(valid_results[valid_results['Performance_Rating'] == 2])
                            st.metric("Below Average", below_avg)
                        
                        # Distribution chart
                        dist_df = valid_results['Performance_Label'].value_counts().reset_index()
                        dist_df.columns = ['Performance', 'Count']
                        
                        fig = px.pie(
                            dist_df,
                            values='Count',
                            names='Performance',
                            title='Performance Distribution',
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="performance_predictions.csv",
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
    **Model Type:** Gradient Boosting Classifier
    
    **Purpose:** Predict employee performance rating (2-4 scale)
    
    **Performance Ratings:**
    - **2**: Below Average
    - **3**: Average
    - **4**: Above Average
    
    **Features Used:**
    - Job Satisfaction (1-4)
    - Environment Satisfaction (1-4)
    - Work-Life Balance (1-4)
    - Monthly Income
    - Years At Company
    - Education Level (1-5)
    - Training Times Last Year
    """)
    
    st.subheader("Input Schema")
    schema = get_input_schema()
    schema_df = pd.DataFrame([
        {
            "Column": col,
            "Type": "Numeric",
            "Description": f"Required input feature"
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
    The model has been trained using:
    - **Algorithm**: Gradient Boosting Classifier
    - **Features**: 7 numerical features
    - **Preprocessing**: StandardScaler normalization
    - **Evaluation**: Accuracy and classification report metrics
    
    The model helps identify:
    - High performers for recognition and promotion
    - Average performers for development opportunities
    - Low performers for performance improvement plans
    """)
    
    st.subheader("Use Cases")
    st.markdown("""
    - **Performance Reviews**: Assist in objective performance assessment
    - **Talent Management**: Identify high-potential employees
    - **Development Planning**: Target training and development initiatives
    - **Compensation Planning**: Align compensation with performance
    - **Retention Strategies**: Identify at-risk performers
    """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Performance Classification Model**
    
    Predicts employee performance rating based on:
    - Job & environment satisfaction
    - Work-life balance
    - Compensation
    - Tenure and experience
    - Education and training
    
    Use this tool to:
    - Assess employee performance objectively
    - Identify development needs
    - Support performance reviews
    - Plan talent management initiatives
    """)
    
    if st.button("Show Example"):
        st.session_state.show_example = True
        example = example_input()
        st.json(example)
