"""
Career Clustering Page
Segments employees into career clusters based on skill composition
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

from inference.skill_clustring_model import (
    SkillClusteringPredictor,
    SKILL_COLUMNS,
    CLUSTER_NAMES,
    CLUSTER_TOP_SKILLS
)

st.set_page_config(
    page_title="Career Clustering",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Employee Skill Clustering")
st.markdown("""
Segment employees into skill-based clusters using their skill composition across 17 technical skills.
This helps identify similar employee profiles for strategic workforce planning and development.
""")

st.markdown("---")

# Load the clustering model
@st.cache_resource
def load_clustering_model():
    try:
        predictor = SkillClusteringPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

predictor = load_clustering_model()

if predictor is None:
    st.error("❌ Clustering model not found. Please ensure the model file exists at Models/skill_composition_kmeans_model.pkl")
    st.stop()

# Use cluster names from the model
CLUSTER_DESCRIPTIONS = {
    cluster_id: {
        "name": CLUSTER_NAMES[cluster_id],
        "description": f"Employees with focus on: {', '.join(CLUSTER_TOP_SKILLS[cluster_id][:3])}",
        "top_skills": CLUSTER_TOP_SKILLS[cluster_id]
    }
    for cluster_id in CLUSTER_NAMES.keys()
}

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Assignment", "Batch Clustering", "Cluster Analysis", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Assign Employee to Skill Cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Skill Ratings")
        st.info("Rate each skill from 1-10 based on proficiency level")
        
        # Create skill input for all 17 skills
        skill_values = {}
        
        # Split skills into two columns for better layout
        col_a, col_b = st.columns(2)
        
        mid_point = len(SKILL_COLUMNS) // 2
        
        with col_a:
            for skill in SKILL_COLUMNS[:mid_point]:
                skill_values[skill] = st.slider(
                    skill,
                    min_value=1,
                    max_value=7,
                    value=5,
                    step=1,
                    key=f"skill_{skill}"
                )
        
        with col_b:
            for skill in SKILL_COLUMNS[mid_point:]:
                skill_values[skill] = st.slider(
                    skill,
                    min_value=1,
                    max_value=7,
                    value=5,
                    step=1,
                    key=f"skill_{skill}"
                )
    
    with col2:
        st.subheader("🎯 Cluster Assignment")
        
        if st.button("🎯 Assign to Cluster", use_container_width=True, type="primary"):
            try:
                # Use the predictor's predict method
                result = predictor.predict_cluster(skill_values, include_details=False)
                
                cluster_id = result['cluster_id']
                cluster_name = result['cluster_name']
                top_skills = result['top_skills']
                
                # Get cluster info
                cluster_info = CLUSTER_DESCRIPTIONS.get(cluster_id, {
                    "name": cluster_name,
                    "description": f"Skill cluster {cluster_id}",
                    "top_skills": top_skills
                })
                
                # Display assignment
                st.success(f"✅ **{cluster_info['name']}**", icon="✅")
                st.write(f"**Description**: {cluster_info['description']}")
                
                # Display top skills for this cluster
                st.write("**Top Skills for this Cluster**:")
                for skill in top_skills[:5]:
                    st.write(f"• {skill}")
                
                # Skill Profile - show top 8 skills as radar
                st.subheader("📊 Your Skill Profile")
                top_8_skills = list(skill_values.keys())[:8]
                top_8_values = [skill_values[s] for s in top_8_skills]
                
                fig = go.Figure(data=[
                    go.Scatterpolar(
                        r=top_8_values,
                        theta=top_8_skills,
                        fill='toself',
                        name='Skill Level',
                        line_color='#4dabf7',
                        fillcolor='rgba(77, 171, 247, 0.3)'
                    )
                ])
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    height=400,
                    showlegend=False,
                    title="Top 8 Skills",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Employee summary - show all skills
                st.subheader("📋 All Skill Ratings")
                summary_df = pd.DataFrame({
                    'Skill': list(skill_values.keys()),
                    'Rating': list(skill_values.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.session_state.last_cluster = cluster_id
                st.session_state.last_cluster_info = cluster_info
                
            except Exception as e:
                st.error(f"Cluster assignment failed: {str(e)}")
        
        # Career development suggestions
        st.subheader("💡 Development Suggestions")
        with st.expander("View recommendations", expanded=False):
            st.info("""
            **Career Development Path:**
            - Identify skill gaps in your cluster profile
            - Pursue targeted training and certifications
            - Seek mentorship within your cluster or adjacent clusters
            - Build cross-functional experiences
            """)

# Tab 2: Batch Clustering
with tab2:
    st.header("Batch Skill Clustering")
    st.write("Upload a CSV file to cluster multiple employees at once.")
    st.info(f"CSV must contain all {len(SKILL_COLUMNS)} skill columns: {', '.join(SKILL_COLUMNS[:3])}...")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} employees")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🎯 Cluster All Employees", type="primary", use_container_width=True):
                try:
                    results_list = []
                    
                    with st.spinner("Clustering employees..."):
                        for idx, row in df.iterrows():
                            try:
                                # Convert row to dict with just skill columns
                                skill_dict = {skill: row[skill] for skill in SKILL_COLUMNS if skill in row}
                                result = predictor.predict(skill_dict)
                                results_list.append({
                                    'Row': idx + 1,
                                    'Cluster_ID': result['cluster_id'],
                                    'Cluster_Name': result['cluster_name']
                                })
                            except Exception as e:
                                results_list.append({
                                    'Row': idx + 1,
                                    'Cluster_ID': None,
                                    'Cluster_Name': f"Error: {str(e)}"
                                })
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results_list)
                    
                    st.success(f"✅ Clustering completed for {len(df)} employees")
                    
                    # Display results
                    st.subheader("Clustering Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("📊 Cluster Statistics")
                    valid_results = results_df[results_df['Cluster_ID'].notna()]
                    cluster_counts = valid_results['Cluster_Name'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cluster distribution
                        fig = px.pie(
                            values=cluster_counts.values,
                            names=cluster_counts.index,
                            title="Employee Distribution by Cluster",
                            hole=0.3,
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster sizes
                        fig = go.Figure(data=[
                            go.Bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                marker_color='#66d9ef'
                            )
                        ])
                        fig.update_layout(
                            title="Employees per Cluster",
                            xaxis_title="Cluster",
                            yaxis_title="Count",
                            showlegend=False,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster details
                    st.subheader("📋 Cluster Composition")
                    for cluster_id in sorted(results_df['Cluster'].unique()):
                        cluster_data = results_df[results_df['Cluster'] == cluster_id]
                        cluster_info = CLUSTER_DESCRIPTIONS.get(cluster_id, {"name": f"Cluster {cluster_id}"})
                        
                        with st.expander(f"{cluster_info['name']} ({len(cluster_data)} employees)"):
                            st.write(cluster_data)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Clustering Results (CSV)",
                        data=csv,
                        file_name="career_clustering_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"Batch clustering failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")

# Tab 3: Cluster Analysis
with tab3:
    st.header("Cluster Analysis & Insights")
    
    st.subheader("🎯 Career Cluster Profiles")
    
    for cluster_id, info in sorted(CLUSTER_DESCRIPTIONS.items()):
        with st.expander(f"{cluster_id + 1}. {info['name']}", expanded=cluster_id == 0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description**: {info['description']}")
                st.write("**Top Skills**:")
                for skill in info['top_skills']:
                    st.write(f"• {skill}")
            
            with col2:
                st.write("**Typical Profile**:")
                st.write("""
                - Specific experience level
                - Skill specialization
                - Career progression path
                - Development opportunities
                """)
    
    st.markdown("---")
    st.subheader("🔍 Cluster Comparison")
    
    st.write("Use this section to compare different clusters and understand the workforce composition.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cluster1 = st.selectbox("Select Cluster 1", range(len(CLUSTER_DESCRIPTIONS)), key="cluster1")
    with col2:
        cluster2 = st.selectbox("Select Cluster 2", range(len(CLUSTER_DESCRIPTIONS)), key="cluster2")
    
    if cluster1 != cluster2:
        info1 = CLUSTER_DESCRIPTIONS[cluster1]
        info2 = CLUSTER_DESCRIPTIONS[cluster2]
        
        comparison_data = {
            'Aspect': ['Name', 'Description'],
            'Cluster 1': [info1['name'], info1['description']],
            'Cluster 2': [info2['name'], info2['description']]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Tab 4: Model Info
with tab4:
    st.header("Model Information")
    
    st.subheader("📊 About This Model")
    st.info("""
    The Career Clustering model uses K-means clustering to segment employees into distinct career paths
    based on their skills, experience, and professional profile. This helps organizations:
    - Identify similar employee profiles
    - Plan workforce development strategies
    - Design targeted training programs
    - Succession planning
    """)
    
    st.subheader("📋 Expected Input Features")
    st.write("""
    The model expects the following features:
    - **Age**: Employee age
    - **YearsAtCompany**: Tenure at organization
    - **TotalExperience**: Total work experience
    - **TechnicalSkill**: Technical proficiency (1-5)
    - **AnalyticalSkill**: Analytical ability (1-5)
    - **CommunicationSkill**: Communication proficiency (1-5)
    - **LeadershipSkill**: Leadership capability (1-5)
    - **DomainExpertise**: Domain knowledge (1-5)
    - **LearningAgility**: Ability to learn quickly (1-5)
    - **PerformanceRating**: Performance rating (1-5)
    - **Certifications**: Number of certifications
    """)
    
    st.subheader("🎯 Cluster Output")
    st.write(f"The model assigns employees to one of {len(CLUSTER_DESCRIPTIONS)} career clusters:")
    
    for cluster_id, info in sorted(CLUSTER_DESCRIPTIONS.items()):
        st.write(f"- **{info['name']}**: {info['description']}")
    
    st.subheader("💼 Use Cases")
    use_cases = [
        "Workforce Segmentation - Understand your employee population composition",
        "Talent Development - Design cluster-specific training and development programs",
        "Succession Planning - Identify key talent in each cluster",
        "Career Pathing - Guide employees through relevant career progression",
        "Organizational Design - Optimize team composition and structure",
        "Retention Strategy - Develop targeted retention programs by cluster"
    ]
    
    for use_case in use_cases:
        st.write(f"✓ {use_case}")
