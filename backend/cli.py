import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import DATA_FILE, OUTPUT_DIR, RANDOM_STATE, MODEL_PATH, SCALER_PATH
from app.ml.preprocess import load_data, clean_data
from app.ml.clustering import evaluate_k, fit_model
from app.ml.personas import generate_personas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='A', choices=['A', 'B'])
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data(DATA_FILE)
    df = clean_data(df)
    
    features_A = ['Unit price', 'Quantity', 'Rating']
    features_B = ['Sales', 'Quantity', 'gross income', 'Rating']
    
    features = features_A if args.features == 'A' else features_B
    logger.info(f"Using feature set {args.features}: {features}")
    
    # Evaluate
    eval_results = evaluate_k(df, features, k_min=2, k_max=8, random_state=RANDOM_STATE)
    best_k = eval_results['best_k_silhouette']
    logger.info(f"Evaluation Results: {eval_results['evaluations']}")
    logger.info(f"Best K by Silhouette: {best_k}, Elbow K: {eval_results['elbow_k']}")
    
    # Plot Elbow & Silhouette
    k_vals = [r['k'] for r in eval_results['evaluations']]
    inertias = [r['inertia'] for r in eval_results['evaluations']]
    silhouettes = [r['silhouette'] for r in eval_results['evaluations']]
    
    plt.figure()
    plt.plot(k_vals, inertias, marker='o')
    plt.title('Elbow Method')
    plt.savefig(OUTPUT_DIR / 'elbow.png', dpi=300)
    plt.close()
    
    plt.figure()
    plt.plot(k_vals, silhouettes, marker='o', color='orange')
    plt.title('Silhouette Scores')
    plt.savefig(OUTPUT_DIR / 'silhouette.png', dpi=300)
    plt.close()
    
    # Fit Model
    fit_res = fit_model(df, features, best_k, RANDOM_STATE, MODEL_PATH, SCALER_PATH)
    labels = fit_res['labels']
    
    df['Cluster'] = labels
    personas = generate_personas(df, labels, features)
    df['Persona'] = df['Cluster'].map(personas)
    
    # PCA Plot
    pca_pts = fit_res['pca_points']
    pca_centroids = fit_res['pca_centroids']
    plt.figure()
    scatter = plt.scatter([p[0] for p in pca_pts], [p[1] for p in pca_pts], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter([c[0] for c in pca_centroids], [c[1] for c in pca_centroids], marker='X', s=200, color='red')
    plt.title('PCA of Clusters')
    plt.savefig(OUTPUT_DIR / 'pca.png', dpi=300)
    plt.close()
    
    # Save clusters
    df.to_csv(OUTPUT_DIR / 'clusters.csv', index=False)
    
    # Metrics
    metrics_df = pd.DataFrame(eval_results['evaluations'])
    metrics_df.to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
    
    logger.info("Pipeline complete. Artifacts saved to output/")

if __name__ == "__main__":
    main()
