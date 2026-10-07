import pytest
import pandas as pd
from app.ml.clustering import evaluate_k, fit_model

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Unit price': [10, 12, 10, 50, 52, 51, 100, 105, 102],
        'Quantity': [1, 2, 1, 5, 6, 5, 10, 11, 10],
        'Rating': [5, 4, 5, 8, 9, 8, 2, 3, 2]
    })

def test_evaluate_k(sample_data):
    features = ['Unit price', 'Quantity', 'Rating']
    res = evaluate_k(sample_data, features, k_min=2, k_max=4)
    assert len(res['evaluations']) == 3
    assert 'best_k_silhouette' in res
    assert 'elbow_k' in res

def test_fit_model(sample_data):
    features = ['Unit price', 'Quantity', 'Rating']
    res = fit_model(sample_data, features, k=3)
    assert len(res['labels']) == 9
    assert len(res['cluster_centers']) == 3
    assert len(res['pca_points']) == 9
    assert len(res['pca_centroids']) == 3
