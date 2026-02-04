#!/usr/bin/env python3
"""
Make Training Data Script

This script processes raw GCMC simulation data and generates training datasets
for the MAPP model. It includes:
1. Loading and merging isotherm and Qst data
2. Calculating selectivity and applying transformations (Arcsinh, Log, Symlog)
3. Generating train/val/test splits (MOF-based or cluster-based)

For MOF clustering, run 02_mof_clustering.py first to generate cluster labels.

Usage:
    # Use existing processed data
    python experiments/03_make_training_data.py --split-type mof
    
    # Reprocess from raw data
    python experiments/03_make_training_data.py --split-type mof --reprocess
    
    # Cluster-based split
    python experiments/03_make_training_data.py --split-type cluster --cluster-file RAC_and_zeo_features_clustered_r1.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    """Symmetric log transform: sign(x) * log10(1 + |x|/threshold)"""
    x = np.asarray(x)
    return np.sign(x) * np.log10(1 + np.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog transform."""
    y = np.asarray(y)
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def map_composition(df):
    """Map MoleculeFraction to CO2Fraction based on GasName."""
    if df["GasName"] == "CO2":
        return round(df["MoleculeFraction"], 4)
    else:
        return round((1 - df["MoleculeFraction"]), 4)


def calc_selectivity(df, col_prefix="AbsLoading"):
    """Calculate CO2/N2 selectivity."""
    if df[f"{col_prefix}N2"] * df["CO2Fraction"] == 0:
        return None
        
    s = df[f"{col_prefix}CO2"] * (1 - df["CO2Fraction"]) / (df[f"{col_prefix}N2"] * df["CO2Fraction"])
    
    if s > 1000 or s < 1e-3:
        return None
    return s


def truncate_ads(x):
    """Ensure adsorption values are non-negative."""
    if pd.isnull(x):
        return None
    elif x <= 0:
        return 0
    else:
        return x


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_isotherm_data(data_dir):
    """Load isotherm data from TSV files."""
    ads_files = list(data_dir.glob("*isotherm*.tsv"))
    if not ads_files:
        raise FileNotFoundError(f"No isotherm files found in {data_dir}")
    
    df_list = []
    for f in ads_files:
        df = pd.read_csv(f, sep="\t")
        df_list.append(df)
    
    df_ads = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_ads)} isotherm data points from {len(ads_files)} files")
    return df_ads


def load_qst_data(data_dir):
    """Load Qst data from TSV files."""
    qst_files = list(data_dir.glob("*Qst*.tsv"))
    if not qst_files:
        raise FileNotFoundError(f"No Qst files found in {data_dir}")
    
    df_list = []
    for f in qst_files:
        df = pd.read_csv(f, sep="\t")
        df_list.append(df)
    
    df_qst = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_qst)} Qst data points from {len(qst_files)} files")
    return df_qst


def process_isotherm_data(df_ads):
    """Process isotherm data: map composition and pivot."""
    # Map CO2 fraction
    df_ads["CO2Fraction"] = df_ads.apply(map_composition, axis=1)
    
    # Pivot CO2 and N2 data
    df_co2 = df_ads[df_ads["GasName"] == "CO2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "AbsLoading", "ExcessLoading"]
    ].rename(columns={"AbsLoading": "AbsLoadingCO2", "ExcessLoading": "ExcessLoadingCO2"})
    
    df_n2 = df_ads[df_ads["GasName"] == "N2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "AbsLoading", "ExcessLoading"]
    ].rename(columns={"AbsLoading": "AbsLoadingN2", "ExcessLoading": "ExcessLoadingN2"})
    
    print(f"Processed isotherm data: CO2={len(df_co2)}, N2={len(df_n2)}")
    return df_co2, df_n2


def process_qst_data(df_qst):
    """Process Qst data: map composition and pivot."""
    # Map CO2 fraction
    df_qst["CO2Fraction"] = df_qst.apply(map_composition, axis=1)
    
    # Pivot CO2 and N2 data
    df_qst_co2 = df_qst[df_qst["GasName"] == "CO2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "Qst"]
    ].rename(columns={"Qst": "QstCO2"})
    
    df_qst_n2 = df_qst[df_qst["GasName"] == "N2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "Qst"]
    ].rename(columns={"Qst": "QstN2"})
    
    print(f"Processed Qst data: CO2={len(df_qst_co2)}, N2={len(df_qst_n2)}")
    return df_qst_co2, df_qst_n2


def merge_isotherm_qst_data(df_co2, df_n2, df_qst_co2, df_qst_n2):
    """Merge isotherm and Qst data, calculate selectivity."""
    # Start with CO2 adsorption data
    df_merged = df_co2.copy()
    
    # Merge N2, QstCO2, and QstN2 data
    for df_i in [df_n2, df_qst_co2, df_qst_n2]:
        df_merged = pd.merge(df_merged, df_i, 
                           on=["MofName", "Pressure[bar]", "CO2Fraction"], 
                           how="outer")
    
    print(f"Merged data shape: {df_merged.shape}")
    
    # Calculate selectivity
    df_merged.insert(7, "AbsLoadingS", df_merged.apply(calc_selectivity, axis=1, col_prefix="AbsLoading"))
    
    print(f"Final merged data shape: {df_merged.shape}")
    return df_merged


def apply_transformations(df):
    """Apply Arcsinh, Log, and Symlog transformations to data columns."""
    # Define columns and their thresholds for transformations
    columns_config = [
        ("Pressure[bar]", 1e-4),
        ("AbsLoadingCO2", 1e-4),
        ("AbsLoadingN2", 1e-4),
        ("AbsLoadingS", 1e-4),
        ("QstCO2", 1),
        ("QstN2", 1)
    ]
    
    for col, threshold in columns_config:
        if col in df.columns:
            df[f"Arcsinh{col}"] = np.arcsinh(df[col])
            df[f"Log{col}"] = np.log10(df[col] + 1e-5)
            df[f"Symlog{col}"] = symlog(df[col], threshold=threshold)
    
    return df





# =============================================================================
# Dataset Split Functions
# =============================================================================

def create_mof_split(df, data_dir, val_size=1000, test_size=1000, seed=0):
    """Create train/val/test split based on MOF names."""
    split_name = f"mof_split_val{val_size}_test{test_size}_seed{seed}_org"
    output_dir = data_dir / split_name
    output_dir.mkdir(exist_ok=True)
    
    all_mofs = df["MofName"].unique()
    np.random.seed(seed)
    np.random.shuffle(all_mofs)
    
    val_mofs = all_mofs[:val_size]
    test_mofs = all_mofs[val_size:val_size + test_size]
    train_mofs = all_mofs[val_size + test_size:]
    
    df_train = df[df["MofName"].isin(train_mofs)]
    df_val = df[df["MofName"].isin(val_mofs)]
    df_test = df[df["MofName"].isin(test_mofs)]
    
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)
    
    print(f"MOF split created: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"Saved to: {output_dir}")
    
    # Create symlinks for graph data
    _create_graph_symlinks(data_dir, output_dir)
    
    return output_dir





def create_cluster_split(df, data_dir, cluster_file, val_clusters=[8], test_clusters=[9, 10, 11], seed=0):
    """Create train/val/test split based on cluster labels from CSV file.
    
    Parameters:
        df: DataFrame with adsorption data
        data_dir: Directory to save split files
        cluster_file: Path to CSV file with MofName and Cluster columns
        val_clusters: List of cluster IDs for validation
        test_clusters: List of cluster IDs for testing
        seed: Random seed (used in output naming)
    """
    # Load cluster labels
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
    
    df_clusters = pd.read_csv(cluster_file)
    if "MofName" not in df_clusters.columns or "Cluster" not in df_clusters.columns:
        raise ValueError("Cluster file must have 'MofName' and 'Cluster' columns")
    
    print(f"Loaded cluster labels for {len(df_clusters)} MOFs")
    
    # Merge cluster labels with data
    df_with_clusters = df.merge(df_clusters[["MofName", "Cluster"]], on="MofName", how="left")
    df_clustered = df_with_clusters.dropna(subset=["Cluster"])
    
    print(f"Samples with cluster labels: {len(df_clustered)} / {len(df)}")
    
    # Create split
    val_size = len(val_clusters)
    test_size = len(test_clusters)
    split_name = f"mof_cluster_split_val{val_size}_test{test_size}_seed{seed}_org"
    output_dir = data_dir / split_name
    output_dir.mkdir(exist_ok=True)
    
    all_clusters = df_clustered["Cluster"].unique()
    train_clusters = [c for c in all_clusters if c not in val_clusters + test_clusters]
    
    df_train = df_clustered[df_clustered["Cluster"].isin(train_clusters)]
    df_val = df_clustered[df_clustered["Cluster"].isin(val_clusters)]
    df_test = df_clustered[df_clustered["Cluster"].isin(test_clusters)]
    
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)
    
    print(f"Cluster split created:")
    print(f"  Train: {len(df_train)} samples from clusters {train_clusters}")
    print(f"  Val: {len(df_val)} samples from clusters {val_clusters}")
    print(f"  Test: {len(df_test)} samples from clusters {test_clusters}")
    print(f"Saved to: {output_dir}")
    
    # Create symlinks for graph data
    _create_graph_symlinks(data_dir, output_dir)
    
    return output_dir


def _create_graph_symlinks(data_dir, output_dir):
    """Create symlinks for graph data directories."""
    for graph_type in ["graphs", "graphs_grids"]:
        src = data_dir / graph_type
        dst = output_dir / graph_type
        if dst.exists():
            os.remove(dst)
        if src.exists():
            os.symlink(src.absolute(), dst.absolute(), target_is_directory=True)





# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function to create training data."""
    # Determine data directory based on source
    if args.source == "tabular":
        # Original tabular data location from notebook
        tabular_data_dir = PROJECT_ROOT.parent / "MOF-MTHNN/data/MOF_diversity/mc_data_tabular"
        data_dir = PROJECT_ROOT / config["data"]["ddmof_data"]
    else:
        data_dir = PROJECT_ROOT / config["data"]["ddmof_data"]
    
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Make Training Data")
    print("=" * 60)
    
    # Check if processed data already exists
    processed_file = data_dir / "id_condition_ads_qst_org_all.csv"
    
    if processed_file.exists() and not args.reprocess:
        print(f"Loading existing processed data from: {processed_file}")
        df_merged = pd.read_csv(processed_file)
    else:
        print("Processing raw data...")
        
        # Load raw data
        if args.source == "tabular" and tabular_data_dir.exists():
            print(f"Loading data from: {tabular_data_dir}")
            df_ads = load_isotherm_data(tabular_data_dir)
            df_qst = load_qst_data(tabular_data_dir)
        else:
            print(f"Loading data from: {data_dir}")
            df_ads = load_isotherm_data(data_dir)
            df_qst = load_qst_data(data_dir)
        
        print(f"Number of unique MOFs in isotherm data: {df_ads['MofName'].nunique()}")
        print(f"Number of unique MOFs in Qst data: {df_qst['MofName'].nunique()}")
        
        # Process data
        df_co2, df_n2 = process_isotherm_data(df_ads)
        df_qst_co2, df_qst_n2 = process_qst_data(df_qst)
        
        # Merge all data
        df_merged = merge_isotherm_qst_data(df_co2, df_n2, df_qst_co2, df_qst_n2)
        
        # Apply transformations
        df_merged = apply_transformations(df_merged)
        
        # Save processed data
        df_merged.to_csv(processed_file, index=False)
        print(f"Saved processed data to: {processed_file}")
    
    print(f"\nDataset shape: {df_merged.shape}")
    print(f"Number of unique MOFs: {df_merged['MofName'].nunique()}")
    
    # Create splits based on split type
    if args.split_type == "mof":
        create_mof_split(df_merged, data_dir, 
                        val_size=args.val_size, 
                        test_size=args.test_size, 
                        seed=args.seed)
    elif args.split_type == "cluster":
        cluster_file = data_dir / args.cluster_file
        create_cluster_split(df_merged, data_dir, cluster_file,
                            val_clusters=args.val_clusters,
                            test_clusters=args.test_clusters,
                            seed=args.seed)
    
    print("\nTraining data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data for MAPP model")
    parser.add_argument("--split-type", type=str, default="mof",
                       choices=["mof", "cluster"],
                       help="Type of data split: mof (random) or cluster (structure-based)")
    parser.add_argument("--val-size", type=int, default=1000,
                       help="Number of MOFs for validation (mof split)")
    parser.add_argument("--test-size", type=int, default=1000,
                       help="Number of MOFs for testing (mof split)")
    parser.add_argument("--cluster-file", type=str, default="RAC_and_zeo_features_clustered_r1.csv",
                       help="CSV file with cluster labels (cluster split)")
    parser.add_argument("--val-clusters", type=int, nargs='+', default=[8],
                       help="Cluster IDs for validation (cluster split)")
    parser.add_argument("--test-clusters", type=int, nargs='+', default=[9, 10, 11],
                       help="Cluster IDs for test set (cluster split)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility")
    parser.add_argument("--reprocess", action="store_true",
                       help="Reprocess raw data even if processed file exists")
    parser.add_argument("--source", type=str, default="local",
                       choices=["local", "tabular"],
                       help="Data source: local (data/ddmof) or tabular (original location)")
    
    args = parser.parse_args()
    main(args)
