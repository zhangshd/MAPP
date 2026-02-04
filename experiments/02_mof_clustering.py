#!/usr/bin/env python3
"""
MOF Clustering Script

This script performs hierarchical clustering on MOF descriptors. It includes:
1. Loading MOF descriptors (RAC and Zeo features)
2. Performing hierarchical clustering with Ward method
3. Evaluating clustering quality (silhouette score)
4. Generating cluster visualizations

For creating cluster-based data splits, use 02_make_training_data.py --split-type cluster

Usage:
    python 02a_mof_clustering.py [--threshold DISTANCE] [--evaluate]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
import yaml

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# =============================================================================
# Descriptor Loading
# =============================================================================

def load_descriptors(data_dir, filename="RAC_and_zeo_features.csv", training_data_file=None):
    """Load MOF descriptors for clustering.
    
    Parameters:
        data_dir: Directory containing the descriptor file
        filename: Name of the descriptor CSV file
        training_data_file: Optional path to training data CSV to filter MOFs
    
    Returns:
        DataFrame with MofName and feature columns
    """
    desc_file = data_dir / filename
    if not desc_file.exists():
        raise FileNotFoundError(f"Descriptor file not found: {desc_file}")
    
    descriptors = pd.read_csv(desc_file)
    print(f"Loaded {len(descriptors)} rows, {descriptors.shape[1]} columns")
    
    # Rename 'name' column to 'MofName' if it exists
    if 'name' in descriptors.columns:
        descriptors.rename(columns={"name": "MofName"}, inplace=True)
    
    # Drop 'cif_file' column if it exists
    if 'cif_file' in descriptors.columns:
        descriptors.drop(columns=["cif_file"], inplace=True)
    
    # Drop rows with any missing values
    descriptors.dropna(inplace=True, axis=0, how="any")
    print(f"After dropping NaN: {len(descriptors)} MOFs")
    
    # Filter by training data MOFs if provided
    if training_data_file is not None:
        training_file = data_dir / training_data_file
        if training_file.exists():
            df_merged = pd.read_csv(training_file)
            mof_list = df_merged["MofName"].unique().tolist()
            descriptors = descriptors[descriptors["MofName"].isin(mof_list)]
            print(f"After filtering by training data: {len(descriptors)} MOFs, {len(mof_list)} unique MOFs in training data")
        else:
            print(f"Warning: Training data file not found: {training_file}")
    
    print(f"Final: {len(descriptors)} MOFs")
    print(f"Number of features: {descriptors.shape[1] - 1}")  # Exclude MofName
    
    return descriptors


# =============================================================================
# Clustering Functions
# =============================================================================

def perform_clustering(descriptors, distance_threshold=290, method='ward'):
    """Perform hierarchical clustering on MOF descriptors.
    
    Parameters:
        descriptors: DataFrame with MofName and feature columns
        distance_threshold: Distance threshold for clustering
        method: Linkage method ('ward', 'complete', 'average', 'single')
    
    Returns:
        clusters: Cluster labels
        Z: Linkage matrix
        features_scaled: Scaled feature matrix
        scaler: Fitted StandardScaler
    """
    # Standardize features
    scaler = StandardScaler()
    features = descriptors.iloc[:, 1:]  # Exclude MofName column
    features_scaled = scaler.fit_transform(features)
    
    # Hierarchical clustering
    Z = linkage(features_scaled, method=method, metric='euclidean')
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    
    n_clusters = len(np.unique(clusters))
    print(f"Created {n_clusters} clusters with distance threshold {distance_threshold}")
    
    # Print cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    for c, count in zip(unique, counts):
        print(f"  Cluster {c}: {count} MOFs")
    
    return clusters, Z, features_scaled, scaler


def find_cluster_centroids(descriptors, clusters, features_scaled):
    """Find the sample closest to each cluster centroid.
    
    Parameters:
        descriptors: DataFrame with MofName and feature columns
        clusters: Cluster labels
        features_scaled: Scaled feature matrix
    
    Returns:
        DataFrame with cluster, mof_idx, and mof_name columns
    """
    unique_clusters = np.unique(clusters)
    centroids = np.array([
        features_scaled[clusters == c].mean(axis=0) 
        for c in unique_clusters
    ])
    
    closest_samples = []
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = clusters == cluster
        cluster_features = features_scaled[cluster_mask]
        distances = cdist(cluster_features, [centroids[i]])
        closest_idx = np.where(cluster_mask)[0][np.argmin(distances)]
        closest_samples.append({
            'cluster': cluster,
            'mof_idx': closest_idx,
            'mof_name': descriptors.iloc[closest_idx]['MofName']
        })
    
    return pd.DataFrame(closest_samples)


# =============================================================================
# Clustering Evaluation
# =============================================================================

def evaluate_clustering_thresholds(descriptors, t_min=50, t_max=500, t_step=20):
    """Evaluate clustering quality across different distance thresholds.
    
    Parameters:
        descriptors: DataFrame with MofName and feature columns
        t_min, t_max, t_step: Range of thresholds to evaluate
    
    Returns:
        DataFrame with threshold, silhouette_score, and n_clusters columns
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(descriptors.iloc[:, 1:])
    Z = linkage(features_scaled, method='ward', metric='euclidean')
    
    t_values = np.arange(t_min, t_max, t_step)
    results = []
    ch_scores = []
    num_clusters = []
    
    for t in t_values:
        clusters = fcluster(Z, t, criterion='distance')
        n_unique = len(np.unique(clusters))
        
        # Skip evaluation if cluster count hasn't changed
        if num_clusters and n_unique == num_clusters[-1]:
            num_clusters.append(n_unique)
            ch_scores.append(ch_scores[-1])
            results.append({
                'threshold': t,
                'silhouette_score': ch_scores[-1],
                'n_clusters': n_unique
            })
            continue
        
        num_clusters.append(n_unique)
        
        if n_unique > 1 and n_unique < len(features_scaled):
            score = silhouette_score(features_scaled, clusters)
        else:
            score = 0
        
        ch_scores.append(score)
        results.append({
            'threshold': t,
            'silhouette_score': score,
            'n_clusters': n_unique
        })
    
    df_results = pd.DataFrame(results)
    
    # Find best threshold
    best_idx = df_results['silhouette_score'].idxmax()
    best_t = df_results.loc[best_idx, 'threshold']
    best_score = df_results.loc[best_idx, 'silhouette_score']
    print(f"\nBest threshold: {best_t}, Silhouette score: {best_score:.3f}")
    
    return df_results


def plot_clustering_evaluation(df_results, output_path):
    """Plot clustering evaluation results.
    
    Parameters:
        df_results: DataFrame from evaluate_clustering_thresholds
        output_path: Path to save the figure
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Use rainbow colormap colors
    cmap = plt.get_cmap('rainbow')
    color1 = cmap(0.15)  # Blue-ish
    color2 = cmap(0.85)  # Orange-ish
    
    ax1.set_xlabel('Distance Threshold', fontsize=14)
    ax1.set_ylabel('Silhouette Score', color=color1, fontsize=14)
    ax1.plot(df_results['threshold'], df_results['silhouette_score'], 
             color=color1, marker='o', linewidth=2, markersize=6, label='Silhouette Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Mark best threshold
    best_idx = df_results['silhouette_score'].idxmax()
    best_t = df_results.loc[best_idx, 'threshold']
    ax1.axvline(best_t, color='red', linestyle='--', linewidth=2, label=f'Best Threshold: {best_t:.0f}')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Clusters', color=color2, fontsize=14)
    ax2.plot(df_results['threshold'], df_results['n_clusters'], 
             color=color2, marker='s', linewidth=2, markersize=6, label='Number of Clusters')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=12)

    # plt.title('Clustering Quality vs Distance Threshold', fontsize=15, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()
    print(f"Saved clustering evaluation plot to: {output_path}")


def plot_dendrogram(Z, clusters, features_scaled, threshold, output_path):
    """Plot dendrogram of hierarchical clustering.
    
    Parameters:
        Z: Linkage matrix
        clusters: Cluster labels
        features_scaled: Scaled feature matrix
        threshold: Distance threshold used for clustering
        output_path: Path to save the figure
    """
    n_clusters = len(np.unique(clusters))

    fig = plt.figure(facecolor="w", figsize=(8, 6))
    plt.xlabel('MOFCount', fontsize=14, fontweight='bold')
    plt.ylabel('Distance', fontsize=14, fontweight='bold')
    
    dendrogram(Z, truncate_mode='lastp', orientation="top", 
               p=n_clusters, labels=["(1)"]*len(features_scaled),
               leaf_font_size=12)
    
    plt.axhline(y=threshold, color='r', linestyle='--')
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    # Add train/val/test regions with rainbow colormap colors
    train_end = 70  # 前7个簇为train
    val_end = 80    # 倒数第3个簇为val
    test_end = 110  # 最后两个簇为test
    
    cmap = plt.get_cmap('rainbow')
    plt.axvspan(xmin, train_end, ymin=0, ymax=threshold/ymax, color=cmap(0.15), alpha=0.25, label='Train')
    plt.axvspan(train_end, val_end, ymin=0, ymax=threshold/ymax, color=cmap(0.5), alpha=0.25, label='Val')
    plt.axvspan(val_end, test_end, ymin=0, ymax=threshold/ymax, color=cmap(0.85), alpha=0.25, label='Test')
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram to: {output_path}")


def plot_cluster_visualization(descriptors, clusters, features_scaled, centroids_df, output_path, method='pca'):
    """Visualize clusters in 2D using PCA or t-SNE.
    
    Parameters:
        descriptors: DataFrame with MofName
        clusters: Cluster labels
        features_scaled: Scaled feature matrix
        centroids_df: DataFrame with cluster centroids info (from find_cluster_centroids)
        output_path: Path to save the figure
        method: 'pca' or 'tsne'
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(features_scaled)
        xlabel = 'PC1'
        ylabel = 'PC2'
    else:
        reducer = TSNE(n_components=2, perplexity=50, learning_rate=200)
        embedding = reducer.fit_transform(features_scaled)
        xlabel = 'PC1'
        ylabel = 'PC2'
    
    # Create DataFrame for plotting
    decomp_df = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    decomp_df['Cluster'] = clusters
    
    # Mark closest samples (centroids)
    decomp_df['Closest'] = 0
    closest_sample_idxs = centroids_df['mof_idx'].values
    decomp_df.loc[closest_sample_idxs, 'Closest'] = 1
    
    # Use rainbow colormap for consistency with other plots
    unique_clusters = decomp_df['Cluster'].unique()
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i / (len(unique_clusters) - 1)) for i in range(len(unique_clusters))]
    palette_dict = {cluster: colors[i] for i, cluster in enumerate(sorted(unique_clusters))}
    
    plt.figure(figsize=(8, 8))
    
    # Plot all clusters with custom labels
    for cluster in sorted(unique_clusters):
        cluster_data = decomp_df[decomp_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                   color=palette_dict[cluster], s=20, alpha=0.8, edgecolor='gray', linewidth=0.5,
                   label=f'Cluster {cluster}')
    
    # plt.title('Cluster Visualization', fontsize=15, fontweight='bold', pad=10)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Plot closest samples (centroids)
    closest_points = decomp_df[decomp_df['Closest'] == 1]
    plt.scatter(closest_points['PC1'], closest_points['PC2'],
                s=50, color=closest_points['Cluster'].map(palette_dict), edgecolor='black', linewidths=2, 
               label='Centroid', alpha=0.8, zorder=5)

    plt.legend(fontsize=13, loc='best', ncol=2)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization to: {output_path}")




# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function for MOF clustering."""
    data_dir = PROJECT_ROOT / config["data"]["ddmof_data"]
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("MOF Clustering")
    print("=" * 60)
    
    # Load descriptors
    print("\nLoading MOF descriptors...")
    training_data_file = "id_condition_ads_qst_org_all.csv" if args.filter_training else None
    descriptors = load_descriptors(data_dir, training_data_file=training_data_file)
    
    # Evaluate thresholds if requested
    if args.evaluate:
        print("\nEvaluating clustering thresholds...")
        df_eval = evaluate_clustering_thresholds(descriptors)
        plot_clustering_evaluation(df_eval, fig_dir / 'clustering_evaluation.png')
        
        # Save evaluation results
        df_eval.to_csv(fig_dir.parent / 'clustering_evaluation.csv', index=False)
    
    # Perform clustering
    print(f"\nPerforming clustering with threshold={args.threshold}...")
    clusters, Z, features_scaled, scaler = perform_clustering(
        descriptors, 
        distance_threshold=args.threshold
    )
    
    # Add cluster labels to descriptors
    descriptors["Cluster"] = clusters
    output_file = data_dir / "RAC_and_zeo_features_clustered_r1_.csv"
    descriptors.to_csv(output_file, index=False)
    print(f"\nSaved clustered descriptors to: {output_file}")
    
    # Find cluster centroids
    centroids_df = find_cluster_centroids(descriptors, clusters, features_scaled)
    centroids_file = data_dir / "cluster_centroids_r1_.csv"
    centroids_df.to_csv(centroids_file, index=False)
    print(f"Saved cluster centroids to: {centroids_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_dendrogram(Z, clusters, features_scaled, args.threshold, 
                   fig_dir / 'clustering_dendrogram.png')
    plot_cluster_visualization(descriptors, clusters, features_scaled, centroids_df,
                              fig_dir / 'cluster_visualization_pca.png', method='pca')
    
    print("\nMOF clustering complete.")
    print(f"\nTo create cluster-based data splits, run:")
    print(f"  python 02_make_training_data.py --split-type cluster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform MOF clustering")
    parser.add_argument("--threshold", type=float, default=290,
                       help="Distance threshold for hierarchical clustering")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate multiple thresholds and plot results")
    parser.add_argument("--filter-training", action="store_true",
                       help="Filter MOFs to only those in training data (id_condition_ads_qst_org_all.csv)")
    
    args = parser.parse_args()
    main(args)
