import sys
import os
import pytest
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shap_explainer import explain_with_shap

def test_explain_with_shap_structure(mock_model, sample_data):
    # Test the structure of the SHAP explanations
    feature_names = ['feature1', 'feature2']
    X = np.array([[1, 2]])
    
    explanations = explain_with_shap(mock_model, X, feature_names)
    
    # Check if explanations is a list
    assert isinstance(explanations, list)
    
    # Check if each explanation has the required keys
    for explanation in explanations:
        assert 'feature' in explanation
        assert 'shap_value' in explanation
        assert 'effect' in explanation
        assert isinstance(explanation['shap_value'], float)
        assert explanation['effect'] in ['increase', 'decrease']

def test_explain_with_shap_values(mock_model, sample_data):
    # Test the values of SHAP explanations
    feature_names = ['feature1', 'feature2']
    X = np.array([[1, 2]])
    
    explanations = explain_with_shap(mock_model, X, feature_names)
    
    # Check if we get explanations for all features
    assert len(explanations) == len(feature_names)
    
    # Check if effects are correctly determined
    for explanation in explanations:
        if explanation['shap_value'] > 0:
            assert explanation['effect'] == 'increase'
        else:
            assert explanation['effect'] == 'decrease'

def test_explain_with_shap_single_sample(mock_model):
    # Test with a single sample
    feature_names = ['feature1']
    X = np.array([[1]])
    
    explanations = explain_with_shap(mock_model, X, feature_names)
    
    # Check if we get one explanation for one feature
    assert len(explanations) == 1
    assert explanations[0]['feature'] == 'feature1' 