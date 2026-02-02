#!/usr/bin/env python3
"""
Working Capacity and Selectivity Comparison Script

This script compares working capacity (WC) and selectivity (S) calculated from
different prediction methods:
1. GCMC Ground Truth
2. MAPP Direct Prediction
3. MAPP-IAST (mixture ML to predict pure-component isotherms, then IAST)
4. MAPP-Pure-IAST (pure-component ML to predict pure-component isotherms, then IAST)

Adsorption condition: 1 bar, CO2 fraction = 0.15
Desorption condition: 0.1 bar, CO2 fraction = 0.9

Working Capacity = q_CO2(adsorption) - q_CO2(desorption)
Selectivity = (y_CO2 / y_N2) / (x_CO2 / x_N2) = (q_CO2 * x_N2) / (q_N2 * x_CO2)

Author: zhangshd
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Default file paths
DEFAULT_MAPP_PRED_FILE = "results/isotherm_prediction_compare_with_gcmc_ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer_version_14.csv"
DEFAULT_IAST_COMPARE_FILE = "results/iast_comparison_all_ads_co2_n2_v4_seed42_extranformerv4_v14.csv"

# PSA-like conditions
# Adsorption: 1 bar, 10% CO2 (typical flue gas, use 0.1 because IAST data available)
ADS_PRESSURE = 1.0  # bar
ADS_CO2_FRAC = 0.15  

# Desorption: 0.01 bar (0.1 bar for VSA), 90% CO2 (enriched)
DES_PRESSURE = 0.01  # bar
DES_CO2_FRAC = 0.9


def load_data(mapp_file: str, iast_file: str):
    """Load and prepare data from CSV files."""
    # Load IAST comparison file (contains all methods)
    iast_df = pd.read_csv(iast_file)
    
    # Load MAPP direct prediction file
    mapp_df = pd.read_csv(mapp_file)
    
    return mapp_df, iast_df


def filter_condition(df: pd.DataFrame, pressure: float, co2_frac: float, 
                     pressure_tol: float = 0.01, frac_tol: float = 0.01) -> pd.DataFrame:
    """Filter dataframe for specific pressure and CO2 fraction."""
    mask = (
        (np.abs(df['Pressure[bar]'] - pressure) < pressure_tol) &
        (np.abs(df['CO2Fraction'] - co2_frac) < frac_tol)
    )
    return df[mask].copy()


def calculate_selectivity(q_co2: float, q_n2: float, x_co2: float) -> float:
    """
    Calculate CO2/N2 selectivity.
    S = (y_CO2 / y_N2) / (x_CO2 / x_N2)
      = (q_CO2 * (1 - x_CO2)) / (q_N2 * x_CO2)
    """
    x_n2 = 1 - x_co2
    if q_n2 < 1e-10 or x_co2 < 1e-10:
        return np.nan
    selectivity = (q_co2 * x_n2) / (q_n2 * x_co2)
    if selectivity > 1e6 or selectivity < 1e-6:
        return np.nan
    return selectivity


def compute_metrics_for_method(
    ads_df: pd.DataFrame, 
    des_df: pd.DataFrame,
    co2_col: str,
    n2_col: str,
    method_name: str
) -> pd.DataFrame:
    """
    Compute working capacity and selectivity for a given prediction method.
    
    Returns a DataFrame with columns:
    - MofName
    - WC_{method}: Working Capacity
    - S_{method}: Selectivity at adsorption condition
    - WC_S_{method}: Product of WC and S
    """
    # Merge adsorption and desorption data on MofName
    merged = ads_df[['MofName', co2_col, n2_col, 'CO2Fraction']].merge(
        des_df[['MofName', co2_col]],
        on='MofName',
        suffixes=('_ads', '_des')
    )
    
    results = []
    for _, row in merged.iterrows():
        mof_name = row['MofName']
        q_co2_ads = row[f'{co2_col}_ads']
        q_n2_ads = row[n2_col]
        q_co2_des = row[f'{co2_col}_des']
        x_co2 = row['CO2Fraction']
        
        # Skip if any value is NaN
        if any(pd.isna([q_co2_ads, q_n2_ads, q_co2_des])):
            continue
        
        # Working Capacity
        wc = q_co2_ads - q_co2_des
        
        # Selectivity at adsorption condition
        selectivity = calculate_selectivity(q_co2_ads, q_n2_ads, x_co2)
        
        # Product
        if pd.notna(wc) and pd.notna(selectivity):
            wc_s = wc * selectivity
        else:
            wc_s = np.nan
        
        results.append({
            'MofName': mof_name,
            f'WC_{method_name}': wc,
            f'S_{method_name}': selectivity,
            f'WC_S_{method_name}': wc_s
        })
    
    return pd.DataFrame(results)


def main(args):
    """Main function to compute and compare working capacity and selectivity."""
    
    print("=" * 80)
    print("Working Capacity and Selectivity Comparison")
    print("=" * 80)
    print(f"\nAdsorption condition: {ADS_PRESSURE} bar, CO2 fraction = {ADS_CO2_FRAC}")
    print(f"Desorption condition: {DES_PRESSURE} bar, CO2 fraction = {DES_CO2_FRAC}")
    print()
    
    # Load data
    print("Loading data...")
    mapp_df, iast_df = load_data(args.mapp_file, args.iast_file)
    print(f"  MAPP prediction file: {len(mapp_df)} rows")
    print(f"  IAST comparison file: {len(iast_df)} rows")
    
    # Filter for adsorption and desorption conditions
    print("\nFiltering for PSA conditions...")
    
    # Adsorption condition (need both MAPP and IAST data)
    iast_ads = filter_condition(iast_df, ADS_PRESSURE, ADS_CO2_FRAC).dropna(
        subset=['AdsCO2', 'AdsN2'], how='any'
    )
    mapp_ads = filter_condition(mapp_df, ADS_PRESSURE, ADS_CO2_FRAC).dropna(
        subset=['AdsCO2', 'AdsN2'], how='any'
    )
    
    # Desorption condition
    iast_des = filter_condition(iast_df, DES_PRESSURE, DES_CO2_FRAC).dropna(
        subset=['AdsCO2'], how='any'
    )
    mapp_des = filter_condition(mapp_df, DES_PRESSURE, DES_CO2_FRAC).dropna(
        subset=['AdsCO2'], how='any'
    )
    
    print(f"  IAST adsorption samples: {len(iast_ads)}")
    print(f"  IAST desorption samples: {len(iast_des)}")
    print(f"  MAPP adsorption samples: {len(mapp_ads)}")
    print(f"  MAPP desorption samples: {len(mapp_des)}")
    
    # Compute metrics for each method
    print("\nComputing metrics for each method...")
    
    # 1. GCMC Ground Truth
    gcmc_metrics = compute_metrics_for_method(
        iast_ads, iast_des, 'AdsCO2', 'AdsN2', 'GCMC'
    )
    print(f"  GCMC: {len(gcmc_metrics)} MOFs")
    
    # 2. MAPP Direct Prediction (use AdsCO2Predicted and AdsN2Predicted from iast_df)
    mapp_ads_pred = iast_ads.dropna(subset=['AdsCO2Predicted', 'AdsN2Predicted'], how='any')
    mapp_des_pred = iast_des.dropna(subset=['AdsCO2Predicted'], how='any')
    mapp_metrics = compute_metrics_for_method(
        mapp_ads_pred, mapp_des_pred, 'AdsCO2Predicted', 'AdsN2Predicted', 'MAPP'
    )
    print(f"  MAPP Direct: {len(mapp_metrics)} MOFs")
    
    # 3. MAPP-IAST (mixture ML + IAST, use AdsCO2MLIast and AdsN2MLIast)
    mapp_iast_ads = iast_ads.dropna(subset=['AdsCO2MLIast', 'AdsN2MLIast'], how='any')
    mapp_iast_des = iast_des.dropna(subset=['AdsCO2MLIast'], how='any')
    mapp_iast_metrics = compute_metrics_for_method(
        mapp_iast_ads, mapp_iast_des, 'AdsCO2MLIast', 'AdsN2MLIast', 'MAPIast'
    )
    print(f"  MAPP-IAST: {len(mapp_iast_metrics)} MOFs")
    
    # 4. MAPP-Pure-IAST (pure-component ML + IAST, use AdsCO2MLPureIast and AdsN2MLPureIast)
    mapp_pure_iast_ads = iast_ads.dropna(subset=['AdsCO2MLPureIast', 'AdsN2MLPureIast'], how='any')
    mapp_pure_iast_des = iast_des.dropna(subset=['AdsCO2MLPureIast'], how='any')
    mapp_pure_iast_metrics = compute_metrics_for_method(
        mapp_pure_iast_ads, mapp_pure_iast_des, 'AdsCO2MLPureIast', 'AdsN2MLPureIast', 'MAPPureIast'
    )
    print(f"  MAPP-Pure-IAST: {len(mapp_pure_iast_metrics)} MOFs")
    
    # 5. GCMC-IAST (GCMC pure-component + IAST, use AdsCO2GCMCIast and AdsN2GCMCIast)
    gcmc_iast_ads = iast_ads.dropna(subset=['AdsCO2GCMCIast', 'AdsN2GCMCIast'], how='any')
    gcmc_iast_des = iast_des.dropna(subset=['AdsCO2GCMCIast'], how='any')
    gcmc_iast_metrics = compute_metrics_for_method(
        gcmc_iast_ads, gcmc_iast_des, 'AdsCO2GCMCIast', 'AdsN2GCMCIast', 'GCMCIast'
    )
    print(f"  GCMC-IAST: {len(gcmc_iast_metrics)} MOFs")
    
    # Merge all metrics
    print("\nMerging metrics from all methods...")
    all_metrics = gcmc_metrics
    for df in [mapp_metrics, mapp_iast_metrics, mapp_pure_iast_metrics, gcmc_iast_metrics]:
        if len(df) > 0:  # Only merge non-empty dataframes
            all_metrics = all_metrics.merge(df, on='MofName', how='inner')
    
    print(f"  Common MOFs across all methods: {len(all_metrics)}")
    
    # Calculate Spearman correlations with GCMC ground truth
    print("\n" + "=" * 80)
    print("Spearman Correlations with GCMC Ground Truth")
    print("=" * 80)
    
    methods = ['MAPP', 'MAPIast', 'MAPPureIast', 'GCMCIast']
    metrics_types = ['WC', 'S', 'WC_S']
    
    results_table = []
    for method in methods:
        row = {'Method': method}
        for metric in metrics_types:
            gcmc_col = f'{metric}_GCMC'
            pred_col = f'{metric}_{method}'
            
            # Filter valid data
            valid_mask = all_metrics[gcmc_col].notna() & all_metrics[pred_col].notna()
            valid_data = all_metrics[valid_mask]
            
            if len(valid_data) > 2:
                spearman_corr, spearman_p = spearmanr(
                    valid_data[gcmc_col], valid_data[pred_col]
                )
                pearson_corr, pearson_p = pearsonr(
                    valid_data[gcmc_col], valid_data[pred_col]
                )
                row[f'{metric}_Spearman'] = spearman_corr
                row[f'{metric}_Pearson'] = pearson_corr
                row[f'{metric}_N'] = len(valid_data)
            else:
                row[f'{metric}_Spearman'] = np.nan
                row[f'{metric}_Pearson'] = np.nan
                row[f'{metric}_N'] = 0
        
        results_table.append(row)
    
    results_df = pd.DataFrame(results_table)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics
    all_metrics.to_csv(output_dir / 'working_capacity_selectivity_comparison.csv', index=False)
    print(f"\nDetailed metrics saved to: {output_dir / 'working_capacity_selectivity_comparison.csv'}")
    
    # Save correlation summary
    results_df.to_csv(output_dir / 'correlation_summary.csv', index=False)
    print(f"Correlation summary saved to: {output_dir / 'correlation_summary.csv'}")
    
    # Generate comparison plots
    if args.plot:
        print("\nGenerating comparison plots...")
        generate_plots(all_metrics, methods, output_dir)
    
    return all_metrics, results_df


def generate_plots(all_metrics: pd.DataFrame, methods: list, output_dir: Path):
    """Generate comparison plots for WC*S metric."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    method_labels = {
        'MAPP': 'MAPP Direct',
        'MAPIast': 'MAPP + IAST',
        'MAPPureIast': 'ML Pure + IAST',
        'GCMCIast': 'GCMC Pure + IAST'
    }
    
    gcmc_col = 'WC_S_GCMC'
    
    for ax, method in zip(axes, methods):
        pred_col = f'WC_S_{method}'
        
        valid_mask = all_metrics[gcmc_col].notna() & all_metrics[pred_col].notna()
        valid_data = all_metrics[valid_mask]
        
        if len(valid_data) > 2:
            x = valid_data[gcmc_col]
            y = valid_data[pred_col]
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=20)
            
            # Add diagonal line
            lims = [
                min(x.min(), y.min()),
                max(x.max(), y.max())
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
            
            # Calculate correlations
            spearman_corr, _ = spearmanr(x, y)
            pearson_corr, _ = pearsonr(x, y)
            
            ax.set_xlabel('GCMC Ground Truth (WC × S)', fontsize=10)
            ax.set_ylabel(f'{method_labels.get(method, method)} (WC × S)', fontsize=10)
            ax.set_title(f'{method_labels.get(method, method)}\n'
                        f'Spearman: {spearman_corr:.3f}, Pearson: {pearson_corr:.3f}',
                        fontsize=11)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wc_selectivity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {output_dir / 'wc_selectivity_comparison.png'}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Working Capacity and Selectivity from different prediction methods'
    )
    parser.add_argument(
        '--mapp-file', 
        type=str, 
        default=DEFAULT_MAPP_PRED_FILE,
        help='Path to MAPP prediction CSV file'
    )
    parser.add_argument(
        '--iast-file', 
        type=str, 
        default=DEFAULT_IAST_COMPARE_FILE,
        help='Path to IAST comparison CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/wc_selectivity_comparison',
        help='Output directory for results'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        default=True,
        help='Generate comparison plots'
    )
    
    args = parser.parse_args()
    main(args)
