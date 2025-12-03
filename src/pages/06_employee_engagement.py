"""
Employee Engagement Clustering Page
Segments employees into engagement clusters based on satisfaction and career factors
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

from inference.employee_engagement_model import (
    predict_engagement_cluster,
    example_input,
    REQUIRED_COLUMNS,
    CLUSTER_DESCRIPTIONS,
    get_input_schema
)

st.set_page_config(
    page_title="Employee Engagement",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Employee Engagement Clustering")
st.markdown("""
Segment employees into engagement clusters based on satisfaction levels, 
career progression, compensation, and involvement to identify engagement patterns 
and develop targeted retention strategies.
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Assignment", "Batch Clustering", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Assign Employee to Engagement Cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Profile")
        
        # Basic Information
        st.write("**Basic Information**")
        age = st.slider("Age", min_value=18, max_value=65, value=35, step=1)
        monthly_income = st.number_input(
            "Monthly Income (USD)",
            min_value=1000, max_value=20000, value=5000, step=100
        )
        job_level = st.slider("Job Level (1-5)", min_value=1, max_value=5, value=2, step=1)
        
        # Career & Tenure
        st.write("**Career & Tenure**")
        years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=5, step=1)
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=50, value=10, step=1)
        promotion_rate = st.slider(
            "Promotion Rate (0-1)",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Ratio of promotions to years at company"
        )
        
        # Department & Role (encoded)
        st.write("**Department & Role**")
        st.info("💡 Select the encoded values for department and job role (0-9)")
        department_enc = st.number_input(
            "Department (encoded)",
            min_value=0, max_value=9, value=1, step=1,
            help="Encoded department: 0=HR, 1=R&D, 2=Sales, etc."
        )
        job_role_enc = st.number_input(
            "Job Role (encoded)",
            min_value=0, max_value=9, value=3, step=1,
            help="Encoded job role: 0=HR, 1=Manager, 2=Sales Rep, etc."
        )
        
        # Engagement Factors
        st.write("**Engagement Factors**")
        job_involvement = st.slider(
            "Job Involvement (1=Low, 4=High)",
            min_value=1, max_value=4, value=3, step=1
        )
        stock_option_level = st.slider(
            "Stock Option Level (0-3)",
            min_value=0, max_value=3, value=1, step=1
        )
    
    with col2:
        st.subheader("🎯 Engagement Cluster Assignment")
        
        # Prepare input data
        input_data = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'JobLevel': job_level,
            'YearsAtCompany': years_at_company,
            'TotalWorkingYears': total_working_years,
            'PromotionRate': promotion_rate,
            'Department_enc': department_enc,
            'JobRole_enc': job_role_enc,
            'JobInvolvement': job_involvement,
            'StockOptionLevel': stock_option_level
        }
        
        if st.button("🎯 Assign to Cluster", use_container_width=True, type="primary"):
            try:
                result = predict_engagement_cluster(input_data)
                
                # Display cluster assignment
                cluster_name = result['cluster_name']
                cluster_desc = result['cluster_description']
                characteristics = result['characteristics']
                
                # Color coding based on cluster
                cluster_id = result['cluster_id']
                if cluster_id == 0:  # Highly Engaged
                    st.success(f"✅ **{cluster_name}**", icon="⭐")
                    color = "green"
                elif cluster_id == 1:  # Disengaged
                    st.warning(f"⚠️ **{cluster_name}**", icon="⚠️")
                    color = "red"
                elif cluster_id == 3:  # New & Enthusiastic
                    st.info(f"🆕 **{cluster_name}**", icon="🌟")
                    color = "blue"
                else:  # Moderately Engaged
                    st.info(f"📊 **{cluster_name}**", icon="ℹ️")
                    color = "orange"
                
                st.write(f"**Description**: {cluster_desc}")
                
                # Display cluster visualization
                st.subheader("📊 Cluster Profile")
                
                # Create radar chart for employee profile
                categories = ['Age', 'Income', 'Job Level', 'Tenure', 'Experience', 'Involvement']
                values = [
                    age / 65,
                    monthly_income / 20000,
                    job_level / 5,
                    years_at_company / 40,
                    total_working_years / 50,
                    job_involvement / 4
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color=color
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Employee Profile",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display characteristics
                st.subheader("✨ Key Characteristics")
                for char in characteristics:
                    st.markdown(f"- {char}")
                
                # Show input summary
                st.subheader("📊 Employee Profile Summary")
                summary_df = pd.DataFrame({
                    'Factor': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Recommendations based on cluster
                st.subheader("💡 Recommendations")
                if cluster_id == 0:
                    st.success("""
                    **Actions for Highly Engaged Employees:**
                    - Recognize and reward high performance
                    - Provide leadership opportunities
                    - Offer mentorship roles
                    - Consider for succession planning
                    """)
                elif cluster_id == 1:
                    st.warning("""
                    **Actions for Disengaged Employees:**
                    - Conduct engagement survey
                    - Provide development opportunities
                    - Address satisfaction concerns
                    - Create performance improvement plan
                    - Consider retention strategies
                    """)
                elif cluster_id == 3:
                    st.info("""
                    **Actions for New & Enthusiastic Employees:**
                    - Provide comprehensive onboarding
                    - Assign mentor or buddy
                    - Offer training and development
                    - Set clear career path expectations
                    - Regular check-ins and feedback
                    """)
                else:
                    st.info("""
                    **Actions for Moderately Engaged Employees:**
                    - Provide growth opportunities
                    - Increase recognition and feedback
                    - Offer skill development programs
                    - Consider role enhancement
                    - Regular career discussions
                    """)
                
                st.session_state.last_prediction = result
                st.session_state.last_input = input_data
                
            except Exception as e:
                st.error(f"Cluster assignment failed: {str(e)}")

# Tab 2: Batch Clustering
with tab2:
    st.header("Batch Engagement Clustering")
    
    st.markdown("""
    Upload a CSV file with multiple employee profiles to assign all to engagement clusters.
    
    **Required columns:**
    - Age (numeric)
    - MonthlyIncome (numeric)
    - JobLevel (1-5)
    - YearsAtCompany (numeric)
    - TotalWorkingYears (numeric)
    - PromotionRate (0-1)
    - Department_enc (encoded integer)
    - JobRole_enc (encoded integer)
    - JobInvolvement (1-4)
    - StockOptionLevel (0-3)
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
            
            if st.button("🚀 Cluster All Employees", use_container_width=True, type="primary"):
                try:
                    from inference.employee_engagement_model import predict_engagement_cluster
                    
                    # Make batch predictions
                    results = []
                    for idx, row in batch_df.iterrows():
                        try:
                            result = predict_engagement_cluster(row.to_dict())
                            results.append({
                                'Row': idx + 1,
                                'Cluster_ID': result['cluster_id'],
                                'Cluster_Name': result['cluster_name'],
                                'Cluster_Description': result['cluster_description']
                            })
                        except Exception as e:
                            results.append({
                                'Row': idx + 1,
                                'Cluster_ID': None,
                                'Cluster_Name': f"Error: {str(e)}",
                                'Cluster_Description': ""
                            })
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Clustering Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📈 Cluster Distribution")
                    valid_results = results_df[results_df['Cluster_ID'].notna()]
                    
                    if len(valid_results) > 0:
                        # Cluster counts
                        cluster_counts = valid_results['Cluster_Name'].value_counts().reset_index()
                        cluster_counts.columns = ['Cluster', 'Count']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = px.pie(
                                cluster_counts,
                                values='Count',
                                names='Cluster',
                                title='Cluster Distribution'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Bar chart
                            fig = px.bar(
                                cluster_counts,
                                x='Cluster',
                                y='Count',
                                title='Employee Count by Cluster'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Cluster summary
                        st.subheader("📋 Cluster Summary")
                        for cluster_name in cluster_counts['Cluster']:
                            count = cluster_counts[cluster_counts['Cluster'] == cluster_name]['Count'].values[0]
                            percentage = (count / len(valid_results)) * 100
                            st.metric(
                                cluster_name,
                                f"{count} employees",
                                delta=f"{percentage:.1f}% of total"
                            )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="engagement_clustering_results.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"Batch clustering failed: {str(e)}")
                    
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")

# Tab 3: Model Info
with tab3:
    st.header("📚 Model Information")
    
    st.subheader("Model Details")
    st.markdown("""
    **Model Type:** K-Means Clustering
    
    **Purpose:** Segment employees into distinct engagement clusters
    
    **Number of Clusters:** 4
    
    **Features Used:**
    - Age and experience
    - Compensation (monthly income)
    - Job level and role
    - Tenure and promotion history
    - Job involvement
    - Stock options
    """)
    
    st.subheader("Cluster Descriptions")
    for cluster_id, info in CLUSTER_DESCRIPTIONS.items():
        with st.expander(f"Cluster {cluster_id}: {info['name']}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown("**Characteristics:**")
            for char in info['characteristics']:
                st.markdown(f"- {char}")
    
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
    The clustering model has been trained using:
    - **Algorithm**: K-Means Clustering
    - **Optimization**: Elbow method and Silhouette score
    - **Preprocessing**: StandardScaler normalization, outlier removal
    - **Validation**: Cluster separation and interpretation
    
    The model helps identify:
    - Highly engaged and loyal employees
    - At-risk or disengaged employees
    - New hires with high potential
    - Employees needing development
    """)
    
    st.subheader("Use Cases")
    st.markdown("""
    - **Retention Planning**: Identify at-risk employees for targeted retention
    - **Talent Development**: Tailor development programs to cluster needs
    - **Engagement Initiatives**: Design cluster-specific engagement strategies
    - **Succession Planning**: Identify highly engaged employees for leadership
    - **Workforce Analytics**: Understand engagement patterns across organization
    - **HR Strategy**: Data-driven decisions for employee experience improvement
    """)
    
    st.subheader("Interpretation Guide")
    st.markdown("""
    **How to interpret cluster assignments:**
    
    1. **Highly Engaged & Loyal**: Your top performers - retain and promote
    2. **Actively Disengaged**: Need immediate intervention - retention risk
    3. **Moderately Engaged**: Solid performers - opportunity for growth
    4. **New & Enthusiastic**: Recent hires - nurture and develop
    
    Use cluster insights to:
    - Prioritize retention efforts
    - Allocate development resources
    - Design targeted engagement programs
    - Plan succession and talent pipelines
    """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Employee Engagement Clustering**
    
    Segments employees based on:
    - Satisfaction and involvement
    - Career progression
    - Compensation and benefits
    - Tenure and experience
    - Role and responsibility
    
    Use this tool to:
    - Identify engagement patterns
    - Target retention strategies
    - Plan development initiatives
    - Support HR decision-making
    """)
    
    st.subheader("Cluster Overview")
    st.markdown("""
    **4 Engagement Clusters:**
    
    🌟 **Highly Engaged & Loyal**
    - High performers
    - Long tenure
    - Strong satisfaction
    
    ⚠️ **Actively Disengaged**
    - Retention risk
    - Low satisfaction
    - Need intervention
    
    📊 **Moderately Engaged**
    - Stable performers
    - Growth potential
    - Average satisfaction
    
    🆕 **New & Enthusiastic**
    - Recent hires
    - High potential
    - Need mentoring
    """)
    
    if st.button("Show Example"):
        st.session_state.show_example = True
        example = example_input()
        st.json(example)
