from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict, Any
import pandas as pd
import os

from app.core.config import DATA_FILE, OUTPUT_DIR, RANDOM_STATE, MODEL_PATH, SCALER_PATH
from app.ml.preprocess import load_data, clean_data, get_summary, get_correlation_matrix, check_outliers
from app.ml.clustering import evaluate_k, fit_model, predict_single
from app.ml.personas import generate_personas
from app.schemas import EvaluateRequest, FitRequest, PredictRequest

router = APIRouter()

# Global in-memory cache for data
global_df = None

def get_data() -> pd.DataFrame:
    global global_df
    if global_df is None:
        try:
            raw_df = load_data(DATA_FILE)
            global_df = clean_data(raw_df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
    return global_df

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/data/summary")
def get_data_summary():
    df = get_data()
    return get_summary(df)

@router.get("/eda/correlation")
def get_correlation():
    df = get_data()
    return get_correlation_matrix(df)

@router.get("/eda/distributions")
def get_distributions():
    df = get_data()
    numeric_df = df.select_dtypes(include=['number'])
    distributions = {}
    for col in numeric_df.columns:
        counts, bins = pd.cut(df[col], bins=10, retbins=True)
        distributions[col] = {
            "bins": bins.tolist(),
            "counts": counts.value_counts(sort=False).tolist()
        }
    return distributions

@router.get("/eda/outliers")
def get_outliers():
    df = get_data()
    return check_outliers(df)

@router.post("/model/evaluate")
def evaluate_model(request: EvaluateRequest):
    df = get_data()
    # Validate features
    for f in request.features:
        if f not in df.columns:
            raise HTTPException(status_code=422, detail=f"Feature {f} not found in dataset.")
    
    if request.k_min > request.k_max:
        raise HTTPException(status_code=400, detail="k_min cannot be greater than k_max.")
        
    res = evaluate_k(df, request.features, request.k_min, request.k_max, RANDOM_STATE)
    return res

@router.post("/model/fit")
def fit_kmeans_model(request: FitRequest):
    df = get_data()
    for f in request.features:
        if f not in df.columns:
            raise HTTPException(status_code=422, detail=f"Feature {f} not found in dataset.")
            
    res = fit_model(df, request.features, request.k, RANDOM_STATE, MODEL_PATH, SCALER_PATH)
    
    # Save clusters to CSV
    df_out = df.copy()
    df_out['Cluster'] = res['labels']
    personas = generate_personas(df_out, res['labels'], request.features)
    df_out['Persona'] = df_out['Cluster'].map(personas)
    df_out.to_csv(OUTPUT_DIR / 'clusters.csv', index=False)
    
    return {
        "cluster_summary": "Model fitted successfully",
        "pca_points": res['pca_points'],
        "pca_centroids": res['pca_centroids'],
        "silhouette_samples": res['silhouette_samples'],
        "cluster_centers": res['cluster_centers'],
        "labels": res['labels']
    }

@router.get("/segments")
def get_segments():
    cluster_file = OUTPUT_DIR / 'clusters.csv'
    if not cluster_file.exists():
        raise HTTPException(status_code=404, detail="Model has not been fitted yet.")
        
    df = pd.read_csv(cluster_file)
    if 'Cluster' not in df.columns:
        raise HTTPException(status_code=500, detail="Cluster data is corrupted.")
        
    segments = []
    clusters = df['Cluster'].unique()
    for c in clusters:
        c_df = df[df['Cluster'] == c]
        segment_data = {
            "cluster_id": int(c),
            "persona": str(c_df['Persona'].iloc[0]),
            "avg_Income": float(c_df['Annual Income (k$)'].mean()) if 'Annual Income (k$)' in c_df.columns else 0.0,
            "avg_Spending": float(c_df['Spending Score (1-100)'].mean()) if 'Spending Score (1-100)' in c_df.columns else 0.0,
            "avg_Age": float(c_df['Age'].mean()) if 'Age' in c_df.columns else 0.0,
            "share_percent": float(len(c_df) / len(df) * 100),
            "top_Branch": str(c_df['Branch'].mode()[0]) if 'Branch' in c_df.columns else "",
            "top_Product_line": str(c_df['Product line'].mode()[0]) if 'Product line' in c_df.columns else "",
            "top_Customer_type": str(c_df['Customer type'].mode()[0]) if 'Customer type' in c_df.columns else "",
            "top_Payment": str(c_df['Payment'].mode()[0]) if 'Payment' in c_df.columns else ""
        }
        segments.append(segment_data)
    return segments

@router.post("/predict")
def predict_cluster(request: PredictRequest):
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise HTTPException(status_code=404, detail="Model has not been fitted yet.")
        
    res = predict_single(request.unit_price, request.quantity, request.rating, MODEL_PATH, SCALER_PATH)
    
    # We need to get the persona for this cluster
    cluster_file = OUTPUT_DIR / 'clusters.csv'
    persona = f"Segment {res['cluster_id']}"
    if cluster_file.exists():
        df = pd.read_csv(cluster_file)
        if 'Cluster' in df.columns and 'Persona' in df.columns:
            matches = df[df['Cluster'] == res['cluster_id']]['Persona']
            if not matches.empty:
                persona = str(matches.iloc[0])
                
    return {
        "cluster_id": res['cluster_id'],
        "persona": persona,
        "distances": res['distances']
    }

@router.get("/export/clusters.csv")
def export_clusters():
    cluster_file = OUTPUT_DIR / 'clusters.csv'
    if not cluster_file.exists():
        raise HTTPException(status_code=404, detail="Clusters file not found. Fit model first.")
    return FileResponse(path=cluster_file, filename="clusters.csv", media_type='text/csv')
