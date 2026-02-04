#!/usr/bin/env python3
"""
Uncertainty Analysis Script

This script analyzes model uncertainty using Latent Space Variance (LSV).
It builds FAISS index trees for fast nearest neighbor search and estimates
prediction uncertainty based on distance to training samples in latent space.

Usage:
    python 04_uncertainty_analysis.py [--model-path PATH] [--k NUM_NEIGHBORS]
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
from pathlib import Path
import yaml

# Optional: FAISS and UMAP (may not be available)
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not installed. Some features will be disabled.")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# =============================================================================
# LSV Calculation Functions
# =============================================================================

def dist_penalty(d, max_value=87.3):
    """
    Calculate the distance penalty with numerical stability.
    
    Parameters:
        d: Distance value
        max_value: Maximum exponent to prevent underflow
    
    Returns:
        Penalty value (exp(-d^2))
    """
    d_squared = d ** 2
    d_squared_clipped = np.clip(d_squared, 0, 100)
    return np.exp(-d_squared_clipped)


def weighted_average(values, weights):
    """Calculate weighted average."""
    return np.sum(values * weights) / np.sum(weights)


def calculate_lsv_from_tree(tree_dic, latent_vectors_test, k=5):
    """
    Calculate Latent Space Variance (LSV) for regression tasks.
    
    Parameters:
        tree_dic: Dictionary containing FAISS tree, training labels, and average distance
        latent_vectors_test: Latent vectors of test samples
        k: Number of nearest neighbors
    
    Returns:
        Array of uncertainty values
    """
    tree = tree_dic["tree"]
    labels_train = tree_dic["labels_train"]
    avg_dist_train = tree_dic["avg_dist_traintrian"]
    
    # Search for k nearest neighbors
    dist_test, ind_test = tree.search(latent_vectors_test.astype(np.float32), k)
    # Normalize distances by average training distance
    dist_test = dist_test / avg_dist_train
    
    uncertainties = []
    for i in range(len(latent_vectors_test)):
        # Get distances and labels of neighbors
        distances = dist_test[i]
        neighbor_labels = labels_train[ind_test[i]]
        
        # Calculate weights based on distance (already normalized)
        weights = np.array([dist_penalty(d) for d in distances])
        
        # Check if weights sum to zero (numerical underflow case)
        weight_sum = np.sum(weights)
        if weight_sum == 0 or np.isnan(weight_sum):
            weights = np.ones_like(weights) / len(weights)
        
        # Calculate weighted variance
        if np.sum(weights) > 0:
            weighted_mean = weighted_average(neighbor_labels, weights)
            variance = weighted_average((neighbor_labels - weighted_mean) ** 2, weights)
        else:
            variance = np.var(neighbor_labels)
        
        uncertainties.append(variance)
    
    return np.array(uncertainties)


# =============================================================================
# Feature Extraction and Visualization
# =============================================================================

def load_latent_features(log_dir, tasks):
    """Load latent features, labels and predictions from model log directory.
    
    The function loads:
    - Latent features from {split}_{task}_last_layer_fea.npz files
    - Labels and predictions from {split}_{task}_predictions.csv files
    
    Returns:
        tuple: (latent_feas, labels, predictions) dictionaries
    """
    latent_feas = {}
    labels = {}
    predictions = {}
    
    for task in tasks:
        for split in ["train", "val", "test"]:
            # Load latent features from npz file
            fea_file = log_dir / f"{split}_{task}_last_layer_fea.npz"
            if fea_file.exists():
                npz_data = np.load(str(fea_file))
                # Extract array from npz (use first key)
                latent_feas[f"{task}_{split}"] = npz_data[list(npz_data.keys())[0]]
            
            # Load labels and predictions from CSV file
            pred_file = log_dir / f"{split}_{task}_predictions.csv"
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                labels[f"{task}_{split}"] = df['GroundTruth'].values
                predictions[f"{task}_{split}"] = df['Predicted'].values
    
    return latent_feas, labels, predictions


def decompose_features(features, method='pca', n_components=2):
    """Decompose high-dimensional features using PCA or t-SNE."""
    if method == 'pca':
        decomposer = PCA(n_components=n_components)
    elif method == 'tsne':
        decomposer = TSNE(n_components=n_components, perplexity=30, n_iter=1000)
    elif method == 'umap' and HAS_UMAP:
        decomposer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return decomposer.fit_transform(features)


def build_uncertainty_trees(latent_feas, labels, tasks, k=5):
    """Build FAISS trees for uncertainty estimation."""
    if not HAS_FAISS:
        raise ImportError("FAISS is required for uncertainty tree building")
    
    uncertainty_trees = {}
    
    for task in tasks:
        train_key = f"{task}_train"
        if train_key not in latent_feas:
            print(f"Warning: No training features for {task}")
            continue
        
        latent_vec_train = latent_feas[train_key].astype(np.float32)
        labels_train = labels[train_key]
        
        d = latent_vec_train.shape[1]
        nlist = 1000  # Matched with notebook value
        
        # Build IVF index
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        
        # Try to use GPU if available
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass  # Fall back to CPU
        
        index.train(latent_vec_train)
        index.add(latent_vec_train)
        index.nprobe = 100
        
        # Calculate average distance for normalization
        dist_train, _ = index.search(latent_vec_train, k=k)
        avg_dist_train = np.mean(dist_train)
        
        uncertainty_trees[task] = {
            "tree": index,
            "labels_train": labels_train,
            "avg_dist_traintrian": avg_dist_train,
            "k": k
        }
        
        print(f"Built uncertainty tree for {task}: {len(latent_vec_train)} samples, avg_dist={avg_dist_train:.4f}")
    
    return uncertainty_trees


def plot_latent_space(latent_feas, labels, tasks, output_dir, method='pca'):
    """Plot latent space visualization colored by target values.
    
    This function matches the notebook implementation:
    1. Concatenate all splits (train, val, test) for each task
    2. Perform PCA/decomposition on all data together
    3. Only plot test split points
    """
    title_map = {
        "SymlogAbsLoadingCO2": r"$\mathbf{symlog_{10}Q_{CO_2}}$",
        "SymlogAbsLoadingN2": r"$\mathbf{symlog_{10}Q_{N_2}}$",
    }
    
    fig = plt.figure(figsize=(8 * len(tasks), 6))
    gs = GridSpec(1, len(tasks), figure=fig, wspace=0.15, hspace=0.1)
    
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, i])
        
        # Step 1: Concatenate all splits for PCA fitting (matching notebook)
        all_features = []
        all_targets = []
        all_splits_info = []
        
        for split in ['train', 'val', 'test']:
            key = f"{task}_{split}"
            if key in latent_feas:
                all_features.append(latent_feas[key])
                all_targets.append(labels[key])
                all_splits_info.extend([split] * len(latent_feas[key]))
        
        if not all_features:
            continue
        
        # Concatenate all data
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Step 2: Perform decomposition on ALL data (train+val+test)
        decomposed_all = decompose_features(all_features, method=method)
        
        # Step 3: Extract only test split for plotting
        test_start_idx = sum(len(latent_feas[f"{task}_{s}"]) for s in ['train', 'val'] if f"{task}_{s}" in latent_feas)
        test_end_idx = test_start_idx + len(latent_feas[f"{task}_test"])
        
        # Plot colored by target values
        scatter = ax.scatter(decomposed_all[:, 0], decomposed_all[:, 1], 
                            c=all_targets, cmap='rainbow', alpha=0.6, s=5)
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', aspect=20, pad=0.05)
        cbar.set_label(title_map.get(task, task), fontsize=12)
        
        ax.set_xlabel(f'{method.upper()}1', fontsize=12)
        ax.set_ylabel(f'{method.upper()}2', fontsize=12)
        ax.set_title("Last-Layer Features", fontsize=14, fontweight='bold')
    
    plt.savefig(output_dir / f'latent_space_{method}_by_targets.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space plot to: {output_dir / f'latent_space_{method}_by_targets.png'}")


def lsv_analysis(task, latent_feas, uncertainty_trees, targets, predictions, k=10, **kwargs):
    """Analyze and plot MAE & Fraction vs LSV Cutoff."""
    
    # Get LSV (Latent Space Variance)
    variances = calculate_lsv_from_tree(
        uncertainty_trees[task], 
        latent_feas[f"{task}_test"].astype(np.float32), k=k
    )
    
    # Scale variances to [0, 1] for x-axis
    # Note: In original notebook, this variable was named 'avg_distances' but contained variances
    scaler = MinMaxScaler()
    scaled_variances = scaler.fit_transform(variances.reshape(-1, 1)).reshape(-1)
    
    # Keyword arguments
    alpha = kwargs.get('alpha', 0.8)
    xmax = kwargs.get('xmax', None)
    ax = kwargs.get('ax', None)
    frac_cutoff = kwargs.get('frac_cutoff', 0.8)
    x_label = kwargs.get('x_label', True)
    y_label_left = kwargs.get('y_label_left', True)
    y_label_right = kwargs.get('y_label_right', True)
    legend = kwargs.get('legend', False)
    tick_font_size = kwargs.get('tick_font_size', 12)
    label_font_size = kwargs.get('label_font_size', 14)
    title = kwargs.get('title', f'MAE & Fraction vs LSV Cutoff [{task}]')
    
    # Rainbow colormap
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i / 9) for i in range(10)]
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Get targets and predictions
    test_targets = targets[f"{task}_test"]
    test_preds = predictions[f"{task}_test"]
    
    # Sweep through cutoffs
    cutoffs = []
    performances_in = []
    performances_out = []
    fracs = []
    r2_scores_in = []
    step = 0.001
    
    for i in range(1, 996, 1):
        cutoff_val = step * i
        mask_in = scaled_variances <= cutoff_val
        mask_out = scaled_variances > cutoff_val
        
        sub_targets = test_targets[mask_in]
        sub_preds = test_preds[mask_in]
        
        if len(sub_targets) == 0:
            continue
            
        sub_targets_out = test_targets[mask_out]
        sub_preds_out = test_preds[mask_out]
        
        cutoffs.append(cutoff_val)
        performances_in.append(metrics.mean_absolute_error(sub_targets, sub_preds))
        
        if len(sub_targets_out) > 0:
            performances_out.append(metrics.mean_absolute_error(sub_targets_out, sub_preds_out))
        else:
            performances_out.append(0)
        
        if len(sub_targets) > 1:
            r2_scores_in.append(metrics.r2_score(sub_targets, sub_preds))
        else:
            r2_scores_in.append(0)
            
        frac = len(sub_targets) / len(test_targets)
        fracs.append(frac)
        
        if frac > frac_cutoff:
            print(f"LSV cutoff={cutoff_val:.3f}, MAE={performances_in[-1]:.4f}, R²={r2_scores_in[-1]:.4f}")
            break
    
    # Full test set MAE
    metric_full = metrics.mean_absolute_error(test_targets, test_preds)
    
    # Plot MAE curves
    sns.scatterplot(x=cutoffs, y=performances_in, ax=ax, marker='D', 
                   label='MAE of samples inside cutoff', alpha=alpha, color=colors[0])
    sns.scatterplot(x=cutoffs, y=performances_out, ax=ax, marker='^', 
                   label='MAE of samples outside cutoff', alpha=alpha, color=colors[1])
    
    if x_label:
        ax.set_xlabel('Uncertainty Cutoff', fontsize=label_font_size+2)
    if y_label_left:
        ax.set_ylabel('Mean Absolute Error', fontsize=label_font_size+2)
    
    # Determine ylim
    y_max = max(performances_out) if performances_out else max(performances_in)
    ax.set_ylim(-0.01, y_max * 1.5)
    
    # Plot fraction on twin axis
    ax2 = ax.twinx()
    sns.scatterplot(x=cutoffs, y=fracs, ax=ax2, color=colors[2], marker='o', 
                   label='Retained data fraction', alpha=alpha)
    if y_label_right:
        ax2.set_ylabel('Fraction', fontsize=label_font_size+2)
    ax2.set_ylim(-0.01, 1.05)
    
    # Set x limits
    if xmax is None:
        xmax = cutoffs[-1] if cutoffs else 1.0
    ax.set_xlim(0, xmax)
    
    # Add horizontal line for full test set MAE
    ax.hlines(y=metric_full, xmin=0, xmax=xmax, colors=colors[-3], 
             linestyles='dashed', label='MAE of full test set', alpha=alpha)
    
    # Add text annotations
    if performances_in:
        ax2.text(0.7, 0.05, "\n".join([
            f'MAE={performances_in[-1]:.3f}',
            "R$^2$" + f'={r2_scores_in[-1]:.3f}',
            'when:',
            'LSV$_{cutoff}$=' + f'{cutoffs[-1]:.3f}',
        ]), transform=ax2.transAxes, fontsize=label_font_size, ha='left', va='bottom', color=colors[-1])
    
    ymin, ymax = ax.get_ylim()
    y_metric_axes = (metric_full - ymin) / (ymax - ymin) + 0.01
    ax.text(0.7, y_metric_axes, "MAE$_{full}$" + f"={metric_full:.3f}", 
           transform=ax.transAxes, fontsize=label_font_size, ha='left', va='bottom', color=colors[-3])
    
    # Legend
    if legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=tick_font_size)
        ax2.legend().set_visible(False)
    else:
        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)
    
    # Title and formatting
    ax.set_title(title, weight='bold', fontsize=label_font_size+2)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax2.tick_params(axis='both', labelsize=tick_font_size)
    
    return ax


def plot_lsv_analysis_figure(tasks, latent_feas, uncertainty_trees, labels, predictions, 
                             output_path, k=5, title_map=None):
    """Create multi-panel LSV analysis figure."""
    if title_map is None:
        title_map = {
            "SymlogAbsLoadingCO2": r"$\mathbf{symlog_{10}Q_{CO_2}}$",
            "SymlogAbsLoadingN2": r"$\mathbf{symlog_{10}Q_{N_2}}$",
            "SymlogAbsLoadingS": r"$\mathbf{symlog_{10}S}$"
        }
    
    fig = plt.figure(figsize=(8 * len(tasks), 6))
    gs = GridSpec(1, len(tasks), figure=fig, wspace=0.35, hspace=0.2)
    
    show_legend = True
    for i, task in enumerate(tasks):
        if task not in uncertainty_trees:
            continue
        lsv_analysis(task, latent_feas, uncertainty_trees, labels, predictions, 
                    k=k, ax=fig.add_subplot(gs[0, i]), frac_cutoff=0.9, 
                    x_label=True, y_label_left=True, y_label_right=True, 
                    legend=show_legend, title=title_map.get(task, task))
        show_legend = False
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved LSV analysis plot to: {output_path}")


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function for uncertainty analysis."""
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Uncertainty Analysis (LSV)")
    print("=" * 60)
    
    # Determine model path
    if args.model_path:
        log_dir = PROJECT_ROOT / args.model_path
    else:
        log_dir = PROJECT_ROOT / config["models"]["MAPP_GMOF"]
    
    if not log_dir.exists():
        print(f"Error: Model directory not found: {log_dir}")
        return
    
    print(f"Analyzing model: {log_dir}")
    
    # Tasks to analyze (only CO2 and N2 have latent features)
    tasks = ["SymlogAbsLoadingCO2", "SymlogAbsLoadingN2"]
    
    # Load latent features, labels and predictions
    print("\nLoading latent features...")
    latent_feas, labels, predictions = load_latent_features(log_dir, tasks)
    
    if not latent_feas:
        print("No latent features found. Run predict.py first.")
        return
    
    print(f"Loaded features for tasks: {list(set(k.split('_')[0] for k in latent_feas.keys()))}")
    
    # Build uncertainty trees
    if HAS_FAISS:
        print(f"\nBuilding uncertainty trees (k={args.k})...")
        uncertainty_trees = build_uncertainty_trees(latent_feas, labels, tasks, k=args.k)
        
        # Fit scalers for test set and save test uncertainties
        for task in tasks:
            if task in uncertainty_trees and f"{task}_test" in latent_feas:
                uncertainties = calculate_lsv_from_tree(
                    uncertainty_trees[task], 
                    latent_feas[f"{task}_test"].astype(np.float32),
                    k=args.k
                )
                scaler = MinMaxScaler()
                scaled_uncertainties = scaler.fit_transform(uncertainties.reshape(-1, 1)).flatten()
                uncertainty_trees[task]["scaler"] = scaler
                uncertainty_trees[task]["test_uncertaintys"] = scaled_uncertainties
                print(f"{task}: uncertainty range [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        
        # Save uncertainty trees
        output_file = log_dir / "uncertainty_trees.pkl"
        
        # Convert GPU index to CPU for saving
        out_dic = {}
        for task_name, task_dict in uncertainty_trees.items():
            out_dic[task_name] = {}
            for k, v in task_dict.items():
                if k == "tree":
                    try:
                        out_dic[task_name][k] = faiss.index_gpu_to_cpu(v)
                    except Exception:
                        out_dic[task_name][k] = v
                else:
                    out_dic[task_name][k] = v
        
        with open(output_file, "wb") as f:
            pickle.dump(out_dic, f)
        print(f"\nSaved uncertainty trees to: {output_file}")
    
    # Plot latent space
    print("\nGenerating latent space visualization...")
    plot_latent_space(latent_feas, labels, tasks, fig_dir, method='pca')
    
    # LSV analysis (predictions already loaded)
    if HAS_FAISS and predictions and uncertainty_trees:
        print("\nGenerating LSV analysis plot...")
        model_name = log_dir.parts[-2] + "_" + log_dir.parts[-1]
        plot_lsv_analysis_figure(
            tasks, latent_feas, uncertainty_trees, labels, predictions,
            fig_dir / f"{model_name}_LSV_calc.png", k=args.k
        )
    
    print("\nUncertainty analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model uncertainty using LSV")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model log directory (relative to project root)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of nearest neighbors for LSV calculation")
    
    args = parser.parse_args()
    main(args)
