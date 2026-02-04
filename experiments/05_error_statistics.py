#!/usr/bin/env python3
"""
Error Statistics Script

This script performs detailed error analysis on model predictions, including:
1. Error distribution by pressure and CO2 fraction
2. Heatmap visualization of prediction errors
3. Statistical analysis of error patterns

Usage:
    python 05_error_statistics.py [--model-path PATH]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn import metrics
from pathlib import Path
import yaml

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# =============================================================================
# Utility Functions
# =============================================================================

def symlog(x, threshold=1e-4):
    """Symmetric log transform."""
    x = np.asarray(x)
    return np.sign(x) * np.log10(1 + np.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog transform."""
    y = np.asarray(y)
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


# =============================================================================
# Error Visualization Functions
# =============================================================================

def draw_heatmap(df, ax=None, title=None, color_col="Error", 
                 pressure_bins=None, co2_fraction_bins=None, cmap='rainbow'):
    """
    Draw heatmap of prediction error by pressure and CO2 fraction.
    
    Parameters:
        df: DataFrame with predictions and conditions
        ax: Matplotlib axis
        title: Plot title
        color_col: Column to use for coloring (e.g., "Error", "Uncertainty")
        pressure_bins: Custom bins for pressure (log10 scale)
        co2_fraction_bins: Custom bins for CO2 fraction
        cmap: Colormap (default: 'rainbow')
    """
    df = df.copy()
    
    # Calculate error if not present
    if color_col == "Error" and "Error" not in df.columns:
        df["Error"] = np.abs(df["Predicted"] - df["GroundTruth"])
    
    # Log transform pressure
    df["Pressure[log10(bar)]"] = np.log10(df["Pressure[bar]"])
    
    # Default bins (22 bins)
    if pressure_bins is None:
        pressure_bins = np.linspace(df["Pressure[log10(bar)]"].min(), 
                                   df["Pressure[log10(bar)]"].max(), 22)
    if co2_fraction_bins is None:
        co2_fraction_bins = np.linspace(df["CO2Fraction"].min(), 
                                       df["CO2Fraction"].max(), 22)
    
    # Bin data
    df["Pressure_bin"] = pd.cut(df["Pressure[log10(bar)]"], bins=pressure_bins)
    df["CO2Fraction_bin"] = pd.cut(df["CO2Fraction"], bins=co2_fraction_bins)
    
    # Get bin left edges for pivot
    df["Pressure_bin_left"] = df["Pressure_bin"].apply(lambda x: x.left if pd.notna(x) else np.nan)
    df["CO2Fraction_bin_left"] = df["CO2Fraction_bin"].apply(lambda x: x.left if pd.notna(x) else np.nan)
    
    # Calculate mean value per bin
    heatmap_data = df.pivot_table(
        index="Pressure_bin_left",
        columns="CO2Fraction_bin_left",
        values=color_col,
        aggfunc='mean',
        observed=False
    )
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6*len(heatmap_data.columns), 6))
    
    # Plot heatmap using seaborn (like original notebook)
    heatmap = sns.heatmap(heatmap_data, cmap=cmap, 
                         cbar_kws={'label': f'Mean {color_col}', 'aspect': 20, 'pad': 0.02}, 
                         ax=ax)
    
    # Set colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(f'Mean {color_col}', fontsize=12)
    
    # Set tick labels (all ticks)
    ax.set_xticks(ticks=np.arange(len(heatmap_data.columns) + 1))
    xtick_labels = list(heatmap_data.columns)
    if len(xtick_labels) > 0:
        step = xtick_labels[1] - xtick_labels[0] if len(xtick_labels) > 1 else 0.05
        xtick_labels.append(xtick_labels[-1] + step)
    ax.set_xticklabels([f"{x:.2f}" for x in xtick_labels], fontsize=10)
    
    ax.set_yticks(ticks=np.arange(len(heatmap_data.index) + 1))
    ytick_labels = list(heatmap_data.index)
    if len(ytick_labels) > 0:
        step = ytick_labels[1] - ytick_labels[0] if len(ytick_labels) > 1 else 0.2
        ytick_labels.append(ytick_labels[-1] + step)
    ax.set_yticklabels([f"{y:.2f}" for y in ytick_labels], fontsize=10)
    
    # Invert y-axis (like original notebook)
    ax.invert_yaxis()
    
    # Set axis labels (like original notebook)
    ax.set_xlabel(r'x$_{CO_2}$', fontsize=14)
    ax.set_ylabel(r'log$_{10}$P (bar)', fontsize=13)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    return ax


def analyze_error_by_condition(df, condition_col, error_col="Error", n_bins=10):
    """
    Analyze error distribution grouped by a condition.
    
    Parameters:
        df: DataFrame with predictions
        condition_col: Column to group by
        error_col: Error column
        n_bins: Number of bins for continuous variables
    """
    df = df.copy()
    
    if error_col not in df.columns:
        df[error_col] = np.abs(df["Predicted"] - df["GroundTruth"])
    
    # Bin if continuous
    if df[condition_col].dtype in [np.float64, np.float32]:
        df[f"{condition_col}_bin"] = pd.cut(df[condition_col], bins=n_bins)
        group_col = f"{condition_col}_bin"
    else:
        group_col = condition_col
    
    # Calculate statistics
    stats = df.groupby(group_col)[error_col].agg(['mean', 'std', 'count', 'median'])
    stats = stats.reset_index()
    
    return stats


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function for error statistics analysis."""
    import pickle
    
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    results_dir = PROJECT_ROOT / config["output"]["results"]
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Error Statistics Analysis")
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
    
    # Model name for output files
    model_name = log_dir.parts[-2] + "_" + log_dir.parts[-1]
    model_display = r"$\mathbf{MAPP\text{-}GMOF}$"
    
    # Tasks and title mapping (bold format)
    tasks = ["SymlogAbsLoadingCO2", "SymlogAbsLoadingN2"]  # Only CO2 and N2 for heatmaps
    title_map = {
        "SymlogAbsLoadingCO2": r"$\mathbf{symlog_{10}Q_{CO_2}}$ (mol/kg)", 
        "SymlogAbsLoadingN2": r"$\mathbf{symlog_{10}Q_{N_2}}$ (mol/kg)", 
        "SymlogAbsLoadingS": r"$\mathbf{symlog_{10}S}$"
    }
    
    # Load predictions
    model_results = {}
    for task in tasks + ["SymlogAbsLoadingS"]:  # Also load S for report
        pred_file = log_dir / f"test_{task}_predictions.csv"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found")
            continue
        
        df = pd.read_csv(pred_file)
        model_results[task] = {
            'target': df['GroundTruth'].values,
            'pred': df['Predicted'].values,
            'df': df
        }
        print(f"Loaded {task}: {len(df)} samples")
    
    if not model_results:
        print("No prediction files found.")
        return
    
    # Custom bins 
    pressure_bins = list(np.arange(-42, 13, 2) / 10)  # -4.2 to 1.2 in 0.2 steps
    co2_fraction_bins = list(np.arange(0, 1.01, 0.05))  # 0 to 1 in 0.05 steps
    
    # Generate combined error heatmap (1 row, 2 columns)
    print("\nGenerating combined error heatmap...")
    fig = plt.figure(figsize=(6 * len(tasks), 6))
    gs = gridspec.GridSpec(1, len(tasks), width_ratios=[1] * len(tasks), wspace=0.32)
    
    for i, task in enumerate(tasks):
        if task not in model_results:
            continue
        df = model_results[task]['df']
        if 'Pressure[bar]' not in df.columns or 'CO2Fraction' not in df.columns:
            continue
        
        ax = fig.add_subplot(gs[0, i])
        draw_heatmap(df, ax=ax, 
                    title=f"{title_map[task]}\n({model_display})",
                    pressure_bins=pressure_bins,
                    co2_fraction_bins=co2_fraction_bins)
    
    plt.savefig(fig_dir / f"{model_name}_test_error_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined error heatmap to: {fig_dir / f'{model_name}_test_error_heatmap.png'}")
    
    # Load uncertainty trees and generate uncertainty heatmap
    uncertainty_file = log_dir / "uncertainty_trees.pkl"
    if uncertainty_file.exists():
        print("\nLoading uncertainty trees...")
        with open(uncertainty_file, "rb") as f:
            uncertainty_trees = pickle.load(f)
        
        print("Generating combined uncertainty heatmap...")
        fig = plt.figure(figsize=(6 * len(tasks), 6))
        gs = gridspec.GridSpec(1, len(tasks), width_ratios=[1] * len(tasks), wspace=0.32)
        
        for i, task in enumerate(tasks):
            if task not in model_results or task not in uncertainty_trees:
                continue
            df = model_results[task]['df'].copy()
            if 'Pressure[bar]' not in df.columns or 'CO2Fraction' not in df.columns:
                continue
            
            # Add uncertainty column
            if "test_uncertaintys" in uncertainty_trees[task]:
                df['Uncertainty'] = uncertainty_trees[task]["test_uncertaintys"]
            else:
                print(f"  Warning: No test_uncertaintys for {task}, skipping")
                continue
            
            ax = fig.add_subplot(gs[0, i])
            draw_heatmap(df, ax=ax, 
                        title=f"{title_map[task]}\n({model_display})",
                        color_col="Uncertainty",
                        pressure_bins=pressure_bins,
                        co2_fraction_bins=co2_fraction_bins)
        
        plt.savefig(fig_dir / f"{model_name}_test_uncertainty_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved combined uncertainty heatmap to: {fig_dir / f'{model_name}_test_uncertainty_heatmap.png'}")
    else:
        print(f"\nWarning: Uncertainty file not found: {uncertainty_file}")
        print("  Run 04_uncertainty_analysis.py first to generate uncertainty trees.")
    
    print("\nError statistics analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction error statistics")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model log directory (relative to project root)")
    
    args = parser.parse_args()
    main(args)
