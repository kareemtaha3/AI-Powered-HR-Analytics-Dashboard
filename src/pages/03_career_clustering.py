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
import joblib
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Career Clustering",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Employee Career Clustering")
st.markdown("""
Segment employees into distinct career clusters based on skill composition and experience.
This helps identify similar employee profiles for strategic workforce planning and development.
""")

st.markdown("---")

# Load the clustering model
@st.cache_resource
def load_clustering_model():
    model_path = Path(__file__).parent.parent.parent / "Models" / "skill_composition_kmeans_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_clustering_model()

if model is None:
    st.error("❌ Clustering model not found. Please ensure the model file exists at Models/skill_composition_kmeans_model.pkl")
    st.stop()

# Cluster descriptions (customize based on your data)
CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Technical Specialists",
        "description": "Employees with deep technical expertise and specialized skills",
        "characteristics": ["High technical skills", "Specialized knowledge", "Expert-level proficiency"]
    },
    1: {
        "name": "Generalists",
        "description": "Well-rounded employees with broad skill sets across multiple domains",
        "characteristics": ["Diverse skills", "Adaptable", "Multi-functional capabilities"]
    },
    2: {
        "name": "Junior Professionals",
        "description": "Early-career employees with foundational skills and growth potential",
        "characteristics": ["Developing skills", "High learning potential", "Entry to mid-level experience"]
    },
    3: {
        "name": "Senior Leaders",
        "description": "Experienced employees with extensive expertise and leadership capabilities",
        "characteristics": ["Advanced expertise", "Leadership skills", "Strategic capabilities"]
    },
    4: {
        "name": "Specialized Analysts",
        "description": "Employees with analytical and specialized domain expertise",
        "characteristics": ["Analytical skills", "Domain expertise", "Focused specialization"]
    }
}

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Assignment", "Batch Clustering", "Cluster Analysis", "Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.header("Assign Employee to Career Cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Input Employee Profile")
        
        # Basic Information
        st.write("**Basic Information**")
        employee_name = st.text_input("Employee Name", "Employee")
        age = st.slider("Age", min_value=20, max_value=65, value=35, step=1)
        department = st.selectbox("Department", ["Sales", "HR", "Technical", "Finance", "Operations", "R&D"])
        
        st.write("**Experience & Tenure**")
        years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=5, step=1)
        total_experience = st.slider("Total Work Experience (years)", min_value=0, max_value=50, value=10, step=1)
        
        st.write("**Skill Levels (1-5 scale)**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            technical_skill = st.slider("Technical Skills", min_value=1, max_value=5, value=3)
            analytical_skill = st.slider("Analytical Skills", min_value=1, max_value=5, value=3)
            communication_skill = st.slider("Communication Skills", min_value=1, max_value=5, value=3)
        
        with col_b:
            leadership_skill = st.slider("Leadership Skills", min_value=1, max_value=5, value=2)
            domain_expertise = st.slider("Domain Expertise", min_value=1, max_value=5, value=3)
            learning_agility = st.slider("Learning Agility", min_value=1, max_value=5, value=3)
        
        st.write("**Performance**")
        performance_rating = st.slider("Performance Rating (1-5)", min_value=1, max_value=5, value=3)
        certifications = st.number_input("Professional Certifications", min_value=0, max_value=10, value=1, step=1)
    
    with col2:
        st.subheader("🎯 Cluster Assignment")
        
        # Prepare input data
        input_data = {
            'Age': age,
            'YearsAtCompany': years_at_company,
            'TotalExperience': total_experience,
            'TechnicalSkill': technical_skill,
            'AnalyticalSkill': analytical_skill,
            'CommunicationSkill': communication_skill,
            'LeadershipSkill': leadership_skill,
            'DomainExpertise': domain_expertise,
            'LearningAgility': learning_agility,
            'PerformanceRating': performance_rating,
            'Certifications': certifications
        }
        
        if st.button("🎯 Assign to Cluster", use_container_width=True, type="primary"):
            try:
                # Prepare data for prediction
                input_df = pd.DataFrame([input_data])
                
                # Predict cluster
                cluster_id = model.predict(input_df)[0]
                
                # Get cluster info
                cluster_info = CLUSTER_DESCRIPTIONS.get(cluster_id, {
                    "name": f"Cluster {cluster_id}",
                    "description": "Distinct career path profile",
                    "characteristics": []
                })
                
                # Display assignment
                st.success(f"✅ **{cluster_info['name']}**", icon="✅")
                st.write(f"**Description**: {cluster_info['description']}")
                
                # Display characteristics
                st.write("**Key Characteristics**:")
                for char in cluster_info['characteristics']:
                    st.write(f"• {char}")
                
                # Skill Profile
                st.subheader("📊 Skill Profile")
                skill_data = {
                    'Skill': ['Technical', 'Analytical', 'Communication', 'Leadership', 'Domain Expertise', 'Learning Agility'],
                    'Level': [technical_skill, analytical_skill, communication_skill, leadership_skill, domain_expertise, learning_agility]
                }
                
                fig = go.Figure(data=[
                    go.Scatterpolar(
                        r=skill_data['Level'],
                        theta=skill_data['Skill'],
                        fill='toself',
                        name='Skill Level'
                    )
                ])
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Employee summary
                st.subheader("📋 Employee Summary")
                summary_df = pd.DataFrame({
                    'Attribute': list(input_data.keys()),
                    'Value': list(input_data.values())
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
    st.header("Batch Career Clustering")
    st.write("Upload a CSV file to cluster multiple employees at once.")
    
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
                    cluster_assignments = []
                    
                    with st.spinner("Clustering employees..."):
                        try:
                            cluster_assignments = model.predict(df)
                        except:
                            st.error("Clustering failed - please ensure all required columns are present")
                            st.stop()
                    
                    # Add results to dataframe
                    results_df = df.copy()
                    results_df['Cluster'] = cluster_assignments
                    results_df['Cluster_Name'] = results_df['Cluster'].apply(
                        lambda x: CLUSTER_DESCRIPTIONS.get(x, {"name": f"Cluster {x}"})["name"]
                    )
                    
                    st.success(f"✅ Clustering completed for {len(df)} employees")
                    
                    # Display results
                    st.subheader("Clustering Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("📊 Cluster Statistics")
                    cluster_counts = results_df['Cluster_Name'].value_counts().sort_index()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cluster distribution
                        fig = px.pie(
                            values=cluster_counts.values,
                            names=cluster_counts.index,
                            title="Employee Distribution by Cluster",
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster sizes
                        fig = go.Figure(data=[
                            go.Bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                marker_color='indianred'
                            )
                        ])
                        fig.update_layout(
                            title="Employees per Cluster",
                            xaxis_title="Cluster",
                            yaxis_title="Count",
                            showlegend=False
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
                st.write("**Key Characteristics**:")
                for char in info['characteristics']:
                    st.write(f"• {char}")
            
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
