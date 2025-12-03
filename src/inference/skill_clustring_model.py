"""
Skill Clustering Model Inference Module

This module provides functions to predict skill composition clusters for employees
based on their skill ratings across 17 different technical skills.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# Define the skill columns (17 features) - must match the training order
SKILL_COLUMNS = [
    'Database Fundamentals',
    'Computer Architecture',
    'Distributed Computing Systems',
    'Cyber Security',
    'Networking',
    'Software Development',
    'Programming Skills',
    'Project Management',
    'Computer Forensics Fundamentals',
    'Technical Communication',
    'AI ML',
    'Software Engineering',
    'Business Analysis',
    'Communication skills',
    'Data Science',
    'Troubleshooting skills',
    'Graphics Designing'
]

# Cluster names (based on the notebook training results)
CLUSTER_NAMES = {
    0: 'AI-Driven Business & Systems Analysis',
    1: 'Communication-Focused Creative Support',
    2: 'Computational Architecture & Data Science',
    3: 'Technical Support & Database Operations',
    4: 'Technical Communication & Digital Forensics',
    5: 'Software Engineering Core Competency',
    6: 'Programming-Focused Technical Track',
    7: 'Network Engineering Competency',
    8: 'Project & Management Leadership',
    9: 'Cybersecurity Engineering'
}

# Top skills for each cluster (for additional context)
CLUSTER_TOP_SKILLS = {
    0: ['Business Analysis', 'AI ML', 'Distributed Computing Systems'],
    1: ['Communication skills', 'Graphics Designing', 'Troubleshooting skills'],
    2: ['Computer Architecture', 'Data Science', 'AI ML'],
    3: ['Troubleshooting skills', 'Graphics Designing', 'Database Fundamentals'],
    4: ['Technical Communication', 'Computer Forensics Fundamentals', 'AI ML'],
    5: ['Software Engineering', 'Troubleshooting skills', 'Graphics Designing'],
    6: ['Programming Skills', 'Graphics Designing', 'Troubleshooting skills'],
    7: ['Networking', 'Graphics Designing', 'Troubleshooting skills'],
    8: ['Project Management', 'Graphics Designing', 'Troubleshooting skills'],
    9: ['Cyber Security', 'Graphics Designing', 'Troubleshooting skills']
}


class SkillClusteringPredictor:
    """
    Predictor class for employee skill composition clustering.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with the trained model.
        
        Args:
            model_path: Path to the pickled K-Means model. If None, uses default path.
        """
        if model_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = base_dir / 'Models' / 'skill_composition_kmeans_model.pkl'
        
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained K-Means model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the model has been trained and saved."
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _transform_input_values(self, skill_values):
        """
        Transform skill ratings from 1-7 scale to 0-6 scale.
        
        Args:
            skill_values: List or array of 17 skill ratings (1-7)
            
        Returns:
            numpy array of transformed values (0-6)
        """
        skill_array = np.array(skill_values)
        
        # Validate input
        if len(skill_array) != 17:
            raise ValueError(
                f"Expected 17 skill values, got {len(skill_array)}. "
                f"Required skills: {SKILL_COLUMNS}"
            )
        
        if not np.all((skill_array >= 1) & (skill_array <= 7)):
            raise ValueError(
                "All skill values must be between 1 and 7 (inclusive). "
                "1=Not Interested, 2=Poor, 3=Beginner, 4=Average, "
                "5=Intermediate, 6=Excellent, 7=Professional"
            )
        
        # Transform from 1-7 to 0-6
        transformed = skill_array - 1
        
        return transformed.astype(float)
    
    def _apply_row_normalization(self, skill_values):
        """
        Apply row-wise normalization to focus on skill composition.
        
        This normalizes the skills relative to the employee's own profile,
        ensuring we focus on the pattern of skills rather than absolute levels.
        
        Args:
            skill_values: Array of skill values (0-6 scale)
            
        Returns:
            numpy array of normalized values
        """
        mean_val = np.mean(skill_values)
        std_val = np.std(skill_values)
        
        # Avoid division by zero
        if std_val > 0:
            normalized = (skill_values - mean_val) / std_val
        else:
            normalized = skill_values - mean_val
        
        return normalized
    
    def predict_cluster(self, skill_values, include_details=True):
        """
        Predict the skill composition cluster for an employee.
        
        Args:
            skill_values: List or dict of 17 skill ratings (1-7 scale)
                         Can be a list/array in the order of SKILL_COLUMNS,
                         or a dict with skill names as keys.
            include_details: If True, returns detailed information about the cluster
            
        Returns:
            dict containing:
                - cluster_id: Cluster number (0-9)
                - cluster_name: Descriptive name of the cluster
                - top_skills: List of top 3 skills for this cluster
                - confidence_score: float (0-1, higher is better)
                - closest_clusters: List of top 3 closest clusters with their info
                - skill_profile: Original skill values (if include_details=True)
                - normalized_profile: Normalized skill values (if include_details=True)
        """
        # Handle dict input
        if isinstance(skill_values, dict):
            skill_values = [skill_values.get(skill, 1) for skill in SKILL_COLUMNS]
        
        # Transform from 1-7 to 0-6
        transformed_values = self._transform_input_values(skill_values)
        
        # Apply row-wise normalization
        normalized_values = self._apply_row_normalization(transformed_values)
        
        # Predict cluster
        cluster_id = self.model.predict(normalized_values.reshape(1, -1))[0]
        
        # Calculate distances to all cluster centers
        distances = self.model.transform(normalized_values.reshape(1, -1))[0]
        
        # Get the top 3 closest clusters
        closest_clusters_indices = np.argsort(distances)[:3]
        closest_clusters = []
        
        for idx in closest_clusters_indices:
            distance = distances[idx]
            closest_clusters.append({
                'cluster_id': int(idx),
                'cluster_name': CLUSTER_NAMES.get(idx, f'Cluster {idx}'),
                'top_skills': CLUSTER_TOP_SKILLS.get(idx, []),
                'distance': float(distance),
                'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity score
            })
        
        # Get primary cluster distance
        confidence_distance = distances[cluster_id]
        
        # Prepare result
        result = {
            'cluster_id': int(cluster_id),
            'cluster_name': CLUSTER_NAMES.get(cluster_id, f'Cluster {cluster_id}'),
            'top_skills': CLUSTER_TOP_SKILLS.get(cluster_id, []),
            'confidence_score': float(1 / (1 + confidence_distance)),  # Convert distance to 0-1 score
            'closest_clusters': closest_clusters  # Top 3 closest clusters
        }
        
        if include_details:
            result['skill_profile'] = {
                skill: int(val) for skill, val in zip(SKILL_COLUMNS, transformed_values)
            }
            result['normalized_profile'] = {
                skill: float(val) for skill, val in zip(SKILL_COLUMNS, normalized_values)
            }
        
        return result
    
    def get_cluster_info(self, cluster_id):
        """
        Get detailed information about a specific cluster.
        
        Args:
            cluster_id: Cluster number (0-9)
            
        Returns:
            dict with cluster information:
                - cluster_id: int
                - cluster_name: str
                - top_skills: list of top 3 skills
                - description: str describing the cluster
        """
        if cluster_id not in CLUSTER_NAMES:
            raise ValueError(f"Invalid cluster_id. Must be between 0 and {len(CLUSTER_NAMES)-1}")
        
        return {
            'cluster_id': cluster_id,
            'cluster_name': CLUSTER_NAMES[cluster_id],
            'top_skills': CLUSTER_TOP_SKILLS[cluster_id],
            'description': f"Employees in this cluster excel in {CLUSTER_NAMES[cluster_id]}"
        }
    
    def get_all_clusters_info(self):
        """
        Get information about all available clusters.
        
        Returns:
            list of dicts, each containing information about a cluster
        """
        return [self.get_cluster_info(i) for i in range(len(CLUSTER_NAMES))]


# Convenience functions for direct use
def predict_employee_cluster(skill_values):
    """
    Predict cluster for a single employee (convenience function).
    
    Args:
        skill_values: List/array/dict of 17 skill ratings (1-7)
        
    Returns:
        Prediction result dictionary with:
            - cluster_id: int (0-9)
            - cluster_name: str
            - top_skills: list of top 3 skills
            - confidence_score: float (0-1)
    """
    predictor = SkillClusteringPredictor()
    return predictor.predict_cluster(skill_values, include_details=False)


def get_cluster_information(cluster_id):
    """
    Get information about a specific cluster (convenience function).
    
    Args:
        cluster_id: Cluster number (0-9)
        
    Returns:
        dict with cluster information
    """
    predictor = SkillClusteringPredictor()
    return predictor.get_cluster_info(cluster_id)


def get_all_clusters():
    """
    Get information about all clusters (convenience function).
    
    Returns:
        list of dicts with information about all 10 clusters
    """
    predictor = SkillClusteringPredictor()
    return predictor.get_all_clusters_info()


# Example usage for testing
if __name__ == "__main__":
    print("=" * 70)
    print("SKILL CLUSTERING MODEL - INFERENCE TEST")
    print("=" * 70)
    
    # Example 1: Software Developer Profile
    print("\n📊 Example 1: Software Developer Profile")
    print("-" * 70)
    
    example_skills = [
        1,  # Database Fundamentals - Average
        1,  # Computer Architecture - Beginner
        1,  # Distributed Computing Systems - Beginner
        1,  # Cyber Security - Average
        1,  # Networking - Average
        1,  # Software Development - Professional
        1,  # Programming Skills - Professional
        1,  # Project Management - Intermediate
        1,  # Computer Forensics Fundamentals - Poor
        1,  # Technical Communication - Intermediate
        7,  # AI ML - Excellent 
        1,  # Software Engineering - Excellent
        1,  # Business Analysis - Beginner
        1,  # Communication skills - Intermediate
        7,  # Data Science - Excellent
        1,  # Troubleshooting skills - Average
        1   # Graphics Designing - Poor
    ]
    
    result = predict_employee_cluster(example_skills)
    
    print(f"\n✓ Primary Cluster:")
    print(f"  Cluster ID: {result['cluster_id']}")
    print(f"  Cluster Name: {result['cluster_name']}")
    print(f"  Confidence Score: {result['confidence_score']:.2%}")
    print(f"  Top Skills: {', '.join(result['top_skills'][:3])}")
    
    print(f"\n✓ Top 3 Closest Clusters:")
    for i, cluster in enumerate(result['closest_clusters'], 1):
        print(f"\n  {i}. {cluster['cluster_name']} (ID: {cluster['cluster_id']})")
        print(f"     Similarity: {cluster['similarity_score']:.2%}")
        print(f"     Top Skills: {', '.join(cluster['top_skills'][:3])}")
    
