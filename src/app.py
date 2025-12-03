"""
HR Analytics Dashboard - Main Application
Multi-page Streamlit app for HR predictive models
"""

import streamlit as st
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-description {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏢 HR Analytics Dashboard")
st.sidebar.markdown("---")

# Navigation pages
pages = {
    "📊 Dashboard": "dashboard",
    "👥 Attrition Prediction": "attrition",
    "📈 Promotion Eligibility": "promotion",
    "🎯 Skill Clustering": "clustering",
    "⭐ Performance Classification": "performance",
    "💰 Salary Prediction": "salary",
    "🎯 Employee Engagement": "engagement"
}

selected_page = st.sidebar.radio("Select Model", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard provides HR analytics and predictive insights using machine learning models."
)

# Main dashboard page
if selected_page == "📊 Dashboard":
    st.markdown("<div class='main-header'>HR Analytics Dashboard</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **HR Analytics Dashboard** - a comprehensive platform for predictive HR analytics 
    and employee insights powered by machine learning.
    """)
    
    # Overview section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Available", "6", "Active")
    with col2:
        st.metric("Features", "7-17", "Per Model")
    with col3:
        st.metric("Predictions", "Real-time", "Instant")
    
    st.markdown("---")
    
    # Models section
    st.header("Available Models")
    
    # Attrition Model
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("👥 Employee Attrition Prediction")
        st.markdown("""
        <div class='model-description'>
        Predict the likelihood of employee attrition using factors such as:
        • Age, tenure, and work experience
        • Job satisfaction and work-life balance
        • Income and distance from home
        • Environment satisfaction and relationship satisfaction
        
        **Use Case**: Identify at-risk employees for retention programs
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Classification\n\n**Output**: Risk Score")
    
    # Promotion Model
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Promotion Eligibility Prediction")
        st.markdown("""
        <div class='model-description'>
        Assess employee promotion eligibility based on:
        • Performance metrics and ratings
        • Training and development
        • Length of service and awards
        • Age, department, and education
        
        **Use Case**: Identify promotion-ready employees for talent development
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Classification\n\n**Output**: Eligibility Score")
    
    # Skill Clustering Model
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Skill Composition Clustering")
        st.markdown("""
        <div class='model-description'>
        Segment employees into skill-based clusters using:
        • 17 technical skill ratings
        • Skill composition patterns
        • Technical specializations
        • Career path analysis
        
        **Use Case**: Identify similar employee skill profiles for strategic planning
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Clustering\n\n**Output**: Cluster Assignment")
    
    # Performance Classification
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("⭐ Performance Classification")
        st.markdown("""
        <div class='model-description'>
        Predict employee performance rating based on:
        • Job and environment satisfaction
        • Work-life balance
        • Monthly income and tenure
        • Education and training
        
        **Use Case**: Assess performance for reviews and development plans
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Classification\n\n**Output**: Performance Rating")
    
    # Salary Prediction
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("💰 Salary Prediction")
        st.markdown("""
        <div class='model-description'>
        Estimate annual developer salary using:
        • Years of professional coding experience
        • Country and location
        • Education level
        • Developer type and primary language
        
        **Use Case**: Benchmark compensation and plan hiring budgets
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Regression\n\n**Output**: Salary Estimate")
    
    # Employee Engagement
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Employee Engagement Clustering")
        st.markdown("""
        <div class='model-description'>
        Segment employees by engagement levels using:
        • Job involvement and satisfaction
        • Career progression and promotion rate
        • Compensation and benefits
        • Tenure and experience
        
        **Use Case**: Target retention and development strategies by engagement level
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Clustering\n\n**Output**: Engagement Cluster")
    
    st.markdown("---")
    
    # Quick Start section
    st.header("Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Attrition Prediction", use_container_width=True):
            st.switch_page("pages/01_attrition_prediction.py")
        if st.button("⭐ Performance Classification", use_container_width=True):
            st.switch_page("pages/04_performance_classification.py")
    
    with col2:
        if st.button("📊 Promotion Eligibility", use_container_width=True):
            st.switch_page("pages/02_promotion_eligibility.py")
        if st.button("💰 Salary Prediction", use_container_width=True):
            st.switch_page("pages/05_salary_prediction.py")
    
    with col3:
        if st.button("🎯 Skill Clustering", use_container_width=True):
            st.switch_page("pages/03_career_clustering.py")
        if st.button("🎯 Employee Engagement", use_container_width=True):
            st.switch_page("pages/06_employee_engagement.py")
    
    st.markdown("---")
    st.info(
        "💡 **Tip**: Use the sidebar navigation to explore different models and make predictions on employee data."
    )

# Route to specific pages
elif selected_page == "👥 Attrition Prediction":
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from inference.attrition_prediction_model import (
            predict_attrition, 
            example_input, 
            REQUIRED_COLUMNS,
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES
        )
        
        # Import and run the attrition page
        exec(open(Path(__file__).parent / "pages" / "01_attrition_prediction.py").read())
    except FileNotFoundError:
        st.error("Attrition prediction page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading attrition prediction page: {e}")

elif selected_page == "📈 Promotion Eligibility":
    try:
        exec(open(Path(__file__).parent / "pages" / "02_promotion_eligibility.py").read())
    except FileNotFoundError:
        st.error("Promotion eligibility page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading promotion eligibility page: {e}")

elif selected_page == "🎯 Skill Clustering":
    try:
        exec(open(Path(__file__).parent / "pages" / "03_career_clustering.py").read())
    except FileNotFoundError:
        st.error("Career clustering page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading career clustering page: {e}")

elif selected_page == "⭐ Performance Classification":
    try:
        exec(open(Path(__file__).parent / "pages" / "04_performance_classification.py").read())
    except FileNotFoundError:
        st.error("Performance classification page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading performance classification page: {e}")

elif selected_page == "💰 Salary Prediction":
    try:
        exec(open(Path(__file__).parent / "pages" / "05_salary_prediction.py").read())
    except FileNotFoundError:
        st.error("Salary prediction page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading salary prediction page: {e}")

elif selected_page == "🎯 Employee Engagement":
    try:
        exec(open(Path(__file__).parent / "pages" / "06_employee_engagement.py").read())
    except FileNotFoundError:
        st.error("Employee engagement page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading employee engagement page: {e}")
