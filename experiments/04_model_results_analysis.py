#!/usr/bin/env python3
"""
Model Results Analysis Script

This script analyzes model prediction results, generates parity plots for
different models (MAPP and Baseline), and compares GMOF vs GCluster splits.

Usage:
    python 03_model_results_analysis.py [--model mapp|cgcnn|all]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from sklearn import metrics
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import yaml

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# =============================================================================
# Evaluation Functions
# =============================================================================

def regression_eval(y_true, y_pred):
    """Calculate regression metrics: MAE, MAPE, R²."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    # MAPE: 只计算非零真值样本
    zero_mask = (y_true != 0)
    if zero_mask.sum() > 0:
        mape = metrics.mean_absolute_percentage_error(y_true[zero_mask], y_pred[zero_mask])
    else:
        mape = np.nan
    
    r2 = metrics.r2_score(y_true, y_pred)
    return {"MAE": mae, "MAPE": mape, "R$^2$": r2}


def density_scatter(x, y, ax=None, is_cbar=False, log_scale=False, s=5, alpha=0.2, **kwargs):
    """Create a scatter plot with points colored by density using rainbow cmap."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Remove NaN values
    x = np.array(x)
    y = np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    # Calculate point density
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones_like(x)
    
    # Sort by density for better visualization
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Create scatter plot with rainbow colormap
    sc = ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap='rainbow', **kwargs)
    
    if is_cbar:
        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax)
        cbar.ax.set_ylabel('Density')
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    return ax, x, y  # 返回x,y用于计算误差


def plot_scatter(targets, predictions, ax=None, metrics=None, is_cbar=False, 
                title=None, log_scale=False, outfile=None, **kwargs):
    """Create parity plot with regression line, metrics, and error distribution inset."""
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    max_value = max(targets.max(), predictions.max())
    min_value = min(targets.min(), predictions.min())
    offset = (max_value - min_value) * 0.05
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot density scatter
    _, x_sorted, y_sorted = density_scatter(targets, predictions, ax=ax, is_cbar=is_cbar, 
                                            log_scale=log_scale, **kwargs)
    
    # Add diagonal line
    ax.plot([min_value, max_value], [min_value, max_value], 'r--', 
           alpha=0.75, zorder=0, linewidth=1.5)
    
    # Set labels and limits
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    ax.set_xlim(min_value - offset, max_value + offset)
    ax.set_ylim(min_value - offset, max_value + offset)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Display metrics
    if metrics:
        text_content = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, 
                fontsize=12, color='red', ha='left', va='top')
    
    #  (inset axes in lower right)
    errors = np.abs(y_sorted - x_sorted)
    inset_ax = inset_axes(ax, width="40%", height="30%", loc='lower right',
                         bbox_to_anchor=(0.05, 0.1, 0.94, 0.94), 
                         bbox_transform=ax.transAxes, borderpad=0)
    inset_ax.patch.set_alpha(0.1)
    sns.histplot(errors, bins=50, color='#0ca7c1', kde=False, alpha=0.4, ax=inset_ax)
    inset_ax.set_title("Error Distribution", fontsize=10)
    inset_ax.set_xlabel("Error", fontsize=8)
    inset_ax.set_ylabel("Frequency", fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Save to file if needed
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=True)
    
    return ax


# =============================================================================
# Model Analysis Functions
# =============================================================================

def analyze_model_predictions(model_dir, tasks, output_dir, model_name="Model"):
    """Analyze test predictions for a single model."""
    results = {}
    
    for task in tasks:
        pred_file = model_dir / f"test_{task}_predictions.csv"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found")
            continue
        
        df = pd.read_csv(pred_file)
        target = df['GroundTruth'].values
        pred = df['Predicted'].values
        
        eval_metrics = regression_eval(target, pred)
        results[task] = {
            'target': target,
            'pred': pred,
            'metrics': eval_metrics
        }
        
        print(f"{model_name} - {task}: MAE={eval_metrics['MAE']:.4f}, "
              f"MAPE={eval_metrics['MAPE']:.4f}, R²={eval_metrics['R$^2$']:.4f}")
    
    return results


def plot_comparison_figure(model_results, title_map, output_path, fig_title="Model Comparison"):
    """Create comparison plot for multiple models."""
    n_models = len(model_results)
    n_tasks = len(list(model_results.values())[0])
    
    fig = plt.figure(figsize=(6 * n_tasks, 6 * n_models))
    gs = gridspec.GridSpec(n_models, n_tasks, wspace=0.25, hspace=0.3)
    
    for i, (model_name, results) in enumerate(model_results.items()):
        for j, (task, data) in enumerate(results.items()):
            ax = fig.add_subplot(gs[i, j])
            
            title = title_map.get(task, task)
            title = f"{title}\n({model_name})"
            plot_scatter(data['target'], data['pred'], 
                        ax=ax, metrics=data['metrics'], 
                        title=title, is_cbar=False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def generate_metrics_table(model_results, output_path):
    """Generate a CSV table with all model metrics."""
    rows = []
    for model_name, results in model_results.items():
        for task, data in results.items():
            row = {
                'Model': model_name,
                'Task': task,
                'MAE': data['metrics']['MAE'],
                'MAPE': data['metrics']['MAPE'],
                'R2': data['metrics']['R$^2$'],
                'N_samples': len(data['target'])
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics table to: {output_path}")
    return df


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function to analyze model results."""
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    results_dir = PROJECT_ROOT / config["output"]["results"]
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Model Results Analysis")
    print("=" * 60)
    
    # Task and title mapping
    tasks = ["SymlogAbsLoadingCO2", "SymlogAbsLoadingN2", "SymlogAbsLoadingS"]
    title_map = {
        "SymlogAbsLoadingCO2": r"$\mathbf{symlog_{10}Q_{CO_2}}$ (mol/kg)", 
        "SymlogAbsLoadingN2": r"$\mathbf{symlog_{10}Q_{N_2}}$ (mol/kg)", 
        "SymlogAbsLoadingS": r"$\mathbf{symlog_{10}S}$"
    }
    
    # Model directories from config
    model_configs = {
        "MAPP-GMOF": config["models"]["MAPP_GMOF"],
        "MAPP-GCluster": config["models"]["MAPP_GCluster"],
        "Baseline-GMOF": config["models"]["CGCNN_GMOF"],
        "Baseline-GCluster": config["models"]["CGCNN_GCluster"],
    }
    
    # Filter models based on args
    if args.model == "mapp":
        model_configs = {k: v for k, v in model_configs.items() if "MAPP" in k}
    elif args.model == "baseline":
        model_configs = {k: v for k, v in model_configs.items() if "Baseline" in k}
    
    # Analyze each model
    all_results = {}
    for model_name, model_path in model_configs.items():
        if model_path is None:
            print(f"Skipping {model_name}: path not configured")
            continue
        
        model_dir = PROJECT_ROOT / model_path
        if not model_dir.exists():
            print(f"Skipping {model_name}: directory not found at {model_dir}")
            continue
        
        print(f"\nAnalyzing {model_name}...")
        results = analyze_model_predictions(model_dir, tasks, fig_dir, model_name)
        if results:
            all_results[model_name] = results
    
    if not all_results:
        print("No model results found to analyze.")
        return
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # GMOF comparison
    gmof_results = {k: v for k, v in all_results.items() if "GMOF" in k}
    if gmof_results:
        plot_comparison_figure(
            gmof_results, title_map, 
            fig_dir / "GMOF_test_predictions_r1.png",
            "GMOF Split Comparison"
        )
    
    # GCluster comparison
    gcluster_results = {k: v for k, v in all_results.items() if "GCluster" in k}
    if gcluster_results:
        plot_comparison_figure(
            gcluster_results, title_map,
            fig_dir / "GCluster_test_predictions_r1.png", 
            "GCluster Split Comparison"
        )
    
    # Generate metrics table
    metrics_df = generate_metrics_table(all_results, results_dir / "model_metrics_summary.csv")
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    print("\nModel results analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model prediction results")
    parser.add_argument("--model", type=str, default="all",
                       choices=["mapp", "baseline", "all"],
                       help="Which model type to analyze")
    
    args = parser.parse_args()
    main(args)
