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
    "🎯 Career Clustering": "clustering"
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
        st.metric("Models Available", "3", "Active")
    with col2:
        st.metric("Features", "13+", "Per Model")
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
        • Tenure and role tenure
        • Salary and compensation
        • Previous promotions and career growth
        
        **Use Case**: Identify promotion-ready employees for talent development
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Classification\n\n**Output**: Eligibility Score")
    
    # Clustering Model
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Career Path Clustering")
        st.markdown("""
        <div class='model-description'>
        Segment employees into career clusters based on:
        • Skill composition and expertise
        • Experience levels and specializations
        • Career trajectories
        • Role distributions
        
        **Use Case**: Identify similar employee profiles for strategic planning
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Type**: Clustering\n\n**Output**: Cluster ID")
    
    st.markdown("---")
    
    # Quick Start section
    st.header("Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Try Attrition Prediction", use_container_width=True):
            st.switch_page("pages/01_attrition_prediction.py")
    
    with col2:
        if st.button("📊 Try Promotion Eligibility", use_container_width=True):
            st.switch_page("pages/02_promotion_eligibility.py")
    
    with col3:
        if st.button("🎯 Try Career Clustering", use_container_width=True):
            st.switch_page("pages/03_career_clustering.py")
    
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

elif selected_page == "🎯 Career Clustering":
    try:
        exec(open(Path(__file__).parent / "pages" / "03_career_clustering.py").read())
    except FileNotFoundError:
        st.error("Career clustering page not found. Please ensure the pages directory is properly set up.")
    except Exception as e:
        st.error(f"Error loading career clustering page: {e}")
