import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def eda_and_preprocessing(df, output_dir):
    # Check dtypes and missing values
    logger.info("Data Types:\n" + str(df.dtypes))
    logger.info("Missing Values:\n" + str(df.isnull().sum()))
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicates: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        
    # Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
    plt.close()
    
    # VIF Check
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    # Add constant for VIF to avoid infinite VIFs if variables are uncentered
    from statsmodels.tools.tools import add_constant
    numeric_df_const = add_constant(numeric_df)
    vif_data["VIF"] = [variance_inflation_factor(numeric_df_const.values, i) for i in range(numeric_df_const.shape[1])]
    logger.info("VIF:\n" + str(vif_data))
    
    # IQR Outliers
    for col in numeric_df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(output_dir / f'boxplot_{col}.png', dpi=300)
        plt.close()
        
    return df

def run_experiment(df, features, experiment_name, output_dir):
    logger.info(f"Running Experiment {experiment_name} with features: {features}")
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal k
    inertias = []
    sil_scores = []
    k_range = range(2, 9)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.title(f'Elbow Method ({experiment_name})')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(output_dir / f'elbow_{experiment_name}.png', dpi=300)
    plt.close()
    
    # Plot Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sil_scores, marker='o', color='orange')
    plt.title(f'Silhouette Scores ({experiment_name})')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig(output_dir / f'silhouette_{experiment_name}.png', dpi=300)
    plt.close()
    
    # Select best k
    best_k = k_range[np.argmax(sil_scores)]
    logger.info(f"Best k for {experiment_name} by silhouette score: {best_k}")
    
    kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_exp = df.copy()
    df_exp['Cluster'] = labels
    
    # Per-sample Silhouette Plot
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    plt.figure(figsize=(8, 6))
    y_lower = 10
    for i in range(best_k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / best_k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    plt.title(f'Silhouette Plot for k={best_k} ({experiment_name})')
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=sil_scores[np.argmax(sil_scores)], color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / f'silhouette_samples_{experiment_name}.png', dpi=300)
    plt.close()
    
    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, color='red', label='Centroids')
    plt.title(f'PCA Plot of Clusters ({experiment_name})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'pca_{experiment_name}.png', dpi=300)
    plt.close()
    
    # Profiling
    profile_cols = features + ['Sales', 'gross income', 'Branch', 'Product line', 'Customer type', 'Payment']
    profile_cols = list(set(profile_cols) & set(df.columns))
    
    numeric_profile = df_exp.groupby('Cluster')[profile_cols].mean(numeric_only=True)
    categorical_profile = {}
    for col in df_exp[profile_cols].select_dtypes(include=['object']):
        categorical_profile[col] = df_exp.groupby('Cluster')[col].value_counts(normalize=True).unstack().fillna(0)
    
    logger.info(f"Numeric Profile for {experiment_name}:\n{numeric_profile}")
    
    numeric_profile.to_csv(output_dir / f'cluster_profile_numeric_{experiment_name}.csv')
    return df_exp, best_k, sil_scores, inertias

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/SuperMarket Analysis.csv')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Please provide the CSV file.")
        return
        
    df = load_data(data_path)
    df = eda_and_preprocessing(df, output_dir)
    
    # Experiment A
    features_a = ['Unit price', 'Quantity', 'Rating']
    df_a, k_a, sil_a, in_a = run_experiment(df, features_a, 'A_Primary', output_dir)
    
    # Experiment B
    features_b = ['Sales', 'Quantity', 'gross income', 'Rating']
    df_b, k_b, sil_b, in_b = run_experiment(df, features_b, 'B_Requested', output_dir)
    
    logger.info("Experiments complete. Check the output directory for results.")

if __name__ == "__main__":
    main()
