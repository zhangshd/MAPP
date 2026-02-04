#!/usr/bin/env python3
"""
Data Analysis Script

This script analyzes the training dataset and generates visualization plots
including pair plots of key adsorption features.

Usage:
    python 01_data_analysis.py
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def symlog(x, threshold=1e-4):
    """
    Symmetric log transform: linear near 0, logarithmic away from 0.
    
    Args:
        x: Input array (can contain 0 and negative values)
        threshold: Transition point between linear and log regions
        
    Returns:
        Transformed array where:
        - symlog(0) = 0
        - |x| < threshold: approximately linear
        - |x| >= threshold: approximately log10(x)
    
    Formula: sign(x) * log10(1 + |x|/threshold)
    """
    x = np.asarray(x)
    return np.sign(x) * np.log10(1 + np.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """
    Inverse of symlog transform.
    
    Args:
        y: Transformed array
        threshold: Same threshold used in symlog()
        
    Returns:
        Original scale array
    """
    y = np.asarray(y)
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def load_data():
    """Load and preprocess the training data."""
    data_dir = PROJECT_ROOT / config["data"]["ddmof_data"]
    df_merged = pd.read_csv(data_dir / "id_condition_ads_qst_org_all.csv")
    return df_merged


def print_data_statistics(df_merged):
    """Print dataset statistics."""
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total number of samples: {df_merged.shape[0]}")
    print(f"Number of samples with adsCO2: {(df_merged['SymlogAbsLoadingCO2'].notna()).sum()}")
    print(f"Number of samples with adsN2: {(df_merged['SymlogAbsLoadingN2'].notna()).sum()}")
    print(f"Number of samples with S: {(df_merged['SymlogAbsLoadingS'].notna()).sum()}")
    print(f"Number of unique MOFs: {df_merged['MofName'].nunique()}")
    print(f"Number of unique conditions:")
    print(df_merged[["Pressure[bar]", "CO2Fraction"]].nunique())
    print("=" * 60)


def create_pairplot(df_merged, fig_dir):
    """Create and save pair plot of key features."""
    
    # Column name mapping for display
    column_map = {
        "SymlogPressure[bar]": r"$\mathbf{symlog_{10}P (bar)}$", 
        "CO2Fraction": r"$\mathbf{x_{CO_2}}$",
        "SymlogAbsLoadingCO2": r"$\mathbf{symlog_{10}Q_{CO_2} (mol/kg)}$", 
        "SymlogAbsLoadingN2": r"$\mathbf{symlog_{10}Q_{N_2} (mol/kg)}$", 
        "SymlogAbsLoadingS": r"$\mathbf{symlog_{10}S_{CO_2/N_2}}$", 
    }
    
    # Rename columns for plotting
    df_plot = df_merged.rename(columns=column_map)
    
    # Set theme
    sns.set_theme(style="white", font_scale=1.1)
    plt.figure(figsize=(8, 8))
    
    # Create pair grid
    g = sns.PairGrid(df_plot[list(column_map.values())], diag_sharey=False)
    
    # Map diagonal and off-diagonal plots
    g.map_diag(sns.histplot, kde=True, color="#0ca7c1")
    g.map_offdiag(plt.hexbin, cmap="rainbow", gridsize=50, mincnt=10, alpha=0.8)
    
    # Tidy ticks
    for ax in g.axes.flatten():
        ax.tick_params(which='both', direction='in')
    
    # Save figure
    output_path = fig_dir / 'dataset_pairplot_r1.png'
    plt.savefig(output_path, dpi=300, transparent=False)
    print(f"Saved pair plot to: {output_path}")
    plt.close()


def main():
    """Main function to run data analysis."""
    # Create output directory
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    df_merged = load_data()
    
    # Print statistics
    print_data_statistics(df_merged)
    
    # Create visualizations
    print("\nGenerating pair plot...")
    create_pairplot(df_merged, fig_dir)
    
    print("\nData analysis complete.")


if __name__ == "__main__":
    main()
