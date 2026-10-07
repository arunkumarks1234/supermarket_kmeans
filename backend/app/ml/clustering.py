import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import joblib
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def evaluate_k(df: pd.DataFrame, features: List[str], k_min: int = 2, k_max: int = 8, random_state: int = 42) -> Dict[str, Any]:
    """Evaluate K-Means for k in range k_min to k_max."""
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    inertias = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        inertia = float(kmeans.inertia_)
        sil_score = float(silhouette_score(X_scaled, labels))
        inertias.append(inertia)
        results.append({"k": k, "inertia": inertia, "silhouette": sil_score})
        
    # Second-difference method for elbow
    second_diffs = np.diff(np.diff(inertias))
    # Add k_min + 1 because diff reduces array size by 1 each time
    elbow_k = int(np.argmax(second_diffs) + k_min + 1) if len(second_diffs) > 0 else k_min
    
    best_k_silhouette = int(max(results, key=lambda x: x['silhouette'])['k'])
    
    return {
        "evaluations": results,
        "best_k_silhouette": best_k_silhouette,
        "elbow_k": elbow_k
    }

def fit_model(df: pd.DataFrame, features: List[str], k: int, random_state: int = 42, model_path: str = None, scaler_path: str = None) -> Dict[str, Any]:
    """Fit final K-Means model, save it, and return results."""
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    
    if model_path and scaler_path:
        joblib.dump(kmeans, model_path)
        joblib.dump(scaler, scaler_path)
        
    n_comp = min(3, len(features))
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    
    sil_samples = silhouette_samples(X_scaled, labels).tolist()
    
    return {
        "labels": labels.tolist(),
        "pca_points": X_pca.tolist(),
        "pca_centroids": centroids_pca.tolist(),
        "silhouette_samples": sil_samples,
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }

def predict_single(unit_price: float, quantity: int, rating: float, model_path: str, scaler_path: str) -> Dict[str, Any]:
    """Predict cluster for a single new sample. Assumes Experiment A features."""
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X_new = pd.DataFrame([[unit_price, quantity, rating]], columns=['Unit price', 'Quantity', 'Rating'])
    X_scaled = scaler.transform(X_new)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    
    distances = [float(np.linalg.norm(X_scaled[0] - centroid)) for centroid in kmeans.cluster_centers_]
    
    return {
        "cluster_id": cluster_id,
        "distances": distances
    }
