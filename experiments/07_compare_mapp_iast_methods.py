#!/usr/bin/env python3
"""
Compare MAPP and IAST Methods Script

This script performs comprehensive comparison between different prediction approaches:
1. GCMC ground truth
2. MAPP direct prediction (mixture model)
3. GCMC-IAST (GCMC pure + IAST)
4. MAPP-IAST (MAPP mixture model pure + IAST)
5. MAPPPure-IAST (MAPP pure model + IAST)

Uses pyGAPS library for IAST calculations.
Generates detailed metrics, comparison tables, and visualizations.

Usage:
    conda run -n mofnn python 07_compare_mapp_iast_methods.py [--model-name NAME] [--pure-model-name NAME]
    
Environment:
    Requires mofnn conda environment with pygaps, sklearn, pandas, numpy, matplotlib, seaborn
"""

import argparse
import traceback
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics
from pathlib import Path
import yaml

# IAST library
try:
    import pygaps as pg
    import pygaps.modelling as pgm
    import pygaps.iast as pgi
    HAS_PYGAPS = True
except ImportError:
    HAS_PYGAPS = False
    print("Warning: pyGAPS not installed. IAST calculations disabled.")

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
    return np.sign(x) * np.log10(1 + np.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog transform."""
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def map_composition(df):
    """Map MoleculeFraction to CO2Fraction."""
    if df["GasName"] == "CO2":
        return round(df["MoleculeFraction"], 4)
    else:
        return round((1 - df["MoleculeFraction"]), 4)


def calc_selectivity(row, co2_col="AdsCO2", n2_col="AdsN2"):
    """Calculate CO2/N2 selectivity."""
    ads_co2 = row[co2_col]
    ads_n2 = row[n2_col]
    co2_frac = row["CO2Fraction"]
    
    if pd.isna(ads_co2) or pd.isna(ads_n2) or co2_frac == 0 or co2_frac == 1:
        return None
    
    s = ads_co2 * (1 - co2_frac) / (ads_n2 * co2_frac + 1e-10)
    
    if s > 1e3 or s < 1e-3:
        return None
    return s


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_gcmc_data():
    """Load and process GCMC simulation data."""
    ads_file = PROJECT_ROOT / config["data"]["ddmof_data"] / "00-isotherm-data-ddmof_extra_test100.tsv"
    df_ads = pd.read_csv(ads_file, sep='\t')
    
    df_ads["CO2Fraction"] = df_ads.apply(map_composition, axis=1)
    
    df_co2 = df_ads[df_ads["GasName"] == "CO2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "AbsLoading"]
    ].rename(columns={"AbsLoading": "AdsCO2"})
    
    df_n2 = df_ads[df_ads["GasName"] == "N2"][
        ["MofName", "CO2Fraction", "Pressure[bar]", "AbsLoading"]
    ].rename(columns={"AbsLoading": "AdsN2"})
    
    df_merged = pd.merge(df_co2, df_n2, 
                        on=["MofName", "CO2Fraction", "Pressure[bar]"], 
                        how="outer")
    
    df_merged["Pressure[bar]"] = df_merged["Pressure[bar]"].apply(
        lambda x: round(x, 3) if x >= 1 else round(x, 6)
    )
    df_merged["CO2Fraction"] = np.round(df_merged["CO2Fraction"], 3)
    
    print(f"Loaded GCMC data: {len(df_merged)} samples, {df_merged['MofName'].nunique()} MOFs")
    return df_merged


def load_ml_predictions(inference_dir, model_name, tasks, dataset="ddmof", epoch=None):
    """Load ML predictions from multiple task files (matching Notebook)."""
    task_pred_dfs = []
    
    for task in tasks:
        if epoch is not None:
            pred_file = inference_dir / dataset / f"{task}_predictions_{model_name}_epoch{epoch}.csv"
        else:
            pred_file = inference_dir / dataset / f"{task}_predictions_{model_name}.csv"
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        
        df_pred = pd.read_csv(pred_file)
        df_pred.rename(columns={k: task+k for k in ["Predicted", "PredictedStd", "Uncertainty"]}, 
                      inplace=True)
        task_pred_dfs.append(df_pred)
    
    # Merge all tasks
    df_pred = task_pred_dfs[0]
    for i in range(1, len(task_pred_dfs)):
        df_pred = pd.merge(df_pred, task_pred_dfs[i], 
                          on=["CifId", "Pressure[bar]", "CO2Fraction"], 
                          how="outer")
    
    df_pred.rename(columns={"CifId": "MofName"}, inplace=True)
    
    # Apply inverse transformations
    theta = 1e-4
    
    for col in df_pred.columns:
        if "Predicted" not in col:
            continue
        
        if "Symlog" in col:
            if col.endswith("Predicted"):
                df_pred[col.replace("SymlogAbsLoading", "Ads")] = symlog_inverse(df_pred[col], threshold=theta)
            elif col.endswith("PredictedStd"):
                df_pred[col.replace("SymlogAbsLoading", "Ads")] = (
                    theta * (10**(np.abs(df_pred[col.replace("Std", "")])) - 1) * 
                    np.sign(df_pred[col.replace("Std", "")])
                )
    
    # Round for matching
    df_pred["Pressure[bar]"] = df_pred["Pressure[bar]"].apply(
        lambda x: round(x, 3) if x >= 1 else round(x, 6)
    )
    df_pred["CO2Fraction"] = np.round(df_pred["CO2Fraction"], 3)
    
    print(f"Loaded predictions: {len(df_pred)} rows")
    return df_pred


# =============================================================================
# IAST Functions
# =============================================================================

def fit_pure_isotherms(mof_group, col_suff=""):
    """Fit pure component isotherms for CO2 and N2.
    
    Args:
        mof_group: DataFrame containing data for a single MOF
        col_suff: Suffix for column names ("", "Predicted", "PredictedPure")
    """
    if not HAS_PYGAPS:
        return None
    
    isotherms_iast_models = []
    
    # Suppress numerical warnings from pyGAPS
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        
        for gas in ['CO2', 'N2']:
            # Extract pure component data
            if gas == 'CO2':
                loading_col = 'AdsCO2' + col_suff
                pure_data = mof_group.loc[mof_group['CO2Fraction'] == 1.0, 
                                        ['Pressure[bar]', loading_col]].sort_values(by='Pressure[bar]').copy()
            else:
                loading_col = 'AdsN2' + col_suff
                pure_data = mof_group.loc[mof_group['CO2Fraction'] == 0.0, 
                                        ['Pressure[bar]', loading_col]].sort_values(by='Pressure[bar]').copy()
            
            pure_data.rename(columns={loading_col: "AbsLoading"}, inplace=True)
            pure_data = pure_data.dropna()
            
            if len(pure_data) < 3:
                return None
            
            try:
                isotherm = pg.PointIsotherm(
                    isotherm_data=pure_data,
                    pressure_key='Pressure[bar]',
                    loading_key='AbsLoading',
                    material=mof_group["MofName"].iloc[0],
                    adsorbate=gas,
                    temperature=25,
                    pressure_mode='absolute',
                    pressure_unit='bar',
                    loading_basis='molar',
                    loading_unit='mmol',
                    material_basis='mass',
                    material_unit='g',
                    temperature_unit='°C'
                )
                
                model_iso = pgm.model_iso(isotherm, 
                                             model=["Henry", "Langmuir", "DSLangmuir"], 
                                             verbose=False)
                isotherms_iast_models.append(model_iso)
            except Exception:
                return None
    
    return isotherms_iast_models if len(isotherms_iast_models) == 2 else None


def calculate_mixture_isotherms(mof_name, isotherms_iast_models, source_type="GCMC"):
    """Calculate mixture isotherms using IAST.
    
    Args:
        mof_name: Name of the MOF
        isotherms_iast_models: List of fitted isotherm models [CO2, N2]
        source_type: String indicating the source ("GCMC", "ML", "MLPure")
    """
    if not HAS_PYGAPS or isotherms_iast_models is None:
        return pd.DataFrame()
    
    pressures = [0.01] + [0.1*i for i in range(1, 10)] + list(range(1, 11))
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15]
    
    results = []
    
    # Pure components
    for total_pressure in pressures:
        results.append([
            mof_name, total_pressure, 0.0,
            None, isotherms_iast_models[1].loading_at(total_pressure)
        ])
        results.append([
            mof_name, total_pressure, 1.0,
            isotherms_iast_models[0].loading_at(total_pressure), None
        ])
    
    # Mixtures
    for frac in fractions:
        for total_pressure in pressures:
            try:
                # Suppress pyGAPS extrapolation warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*extrapolate.*')
                    warnings.filterwarnings('ignore', message='.*highest pressure.*')
                    mix_ads = pgi.iast_point_fraction(
                        isotherms_iast_models,
                        [frac, 1-frac],
                        total_pressure,
                        verbose=False
                    )
                results.append([mof_name, total_pressure, frac] + mix_ads.tolist())
            except Exception:
                continue
    
    df = pd.DataFrame(results, columns=[
        'MofName', 'Pressure[bar]', 'CO2Fraction', 
        f'AdsCO2{source_type}Iast', f'AdsN2{source_type}Iast'
    ])
    
    df["Pressure[bar]"] = df["Pressure[bar]"].apply(
        lambda x: round(x, 3) if x >= 1 else round(x, 6)
    )
    df["CO2Fraction"] = np.round(df["CO2Fraction"], 3)
    
    return df


def calculate_all_iast(df_compare):
    """Calculate IAST predictions for GCMC, ML, and MLPure."""
    if not HAS_PYGAPS:
        print("pyGAPS not available. Skipping IAST calculations.")
        return df_compare
    
    all_iast_results_gcmc = []
    all_iast_results_ml = []
    all_iast_results_ml_pure = []
    
    mof_list = df_compare["MofName"].unique()
    print(f"\nCalculating IAST for {len(mof_list)} MOFs...")
    
    for i, mof in enumerate(mof_list):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(mof_list)}")
        
        mof_group = df_compare[df_compare["MofName"] == mof]
        
        # GCMC-based IAST
        try:
            isotherms_models = fit_pure_isotherms(mof_group, col_suff="")
            if isotherms_models:
                iast_results = calculate_mixture_isotherms(mof, isotherms_models, "GCMC")
                if len(iast_results) > 0:
                    all_iast_results_gcmc.append(iast_results)
        except Exception:
            pass
        
        # ML-based IAST (from mixture model)
        try:
            isotherms_models = fit_pure_isotherms(mof_group, col_suff="Predicted")
            if isotherms_models:
                iast_results = calculate_mixture_isotherms(mof, isotherms_models, "ML")
                if len(iast_results) > 0:
                    all_iast_results_ml.append(iast_results)
        except Exception:
            pass
        
        # MLPure-IAST (from pure component model)
        try:
            isotherms_models = fit_pure_isotherms(mof_group, col_suff="PredictedPure")
            if isotherms_models:
                iast_results = calculate_mixture_isotherms(mof, isotherms_models, "MLPure")
                if len(iast_results) > 0:
                    all_iast_results_ml_pure.append(iast_results)
        except Exception:
            pass
    
    # Merge all IAST results
    df_result = df_compare.copy()
    
    if all_iast_results_gcmc:
        df_iast_gcmc = pd.concat(all_iast_results_gcmc, ignore_index=True)
        df_result = df_result.merge(
            df_iast_gcmc,
            on=['MofName', 'Pressure[bar]', 'CO2Fraction'],
            how='left'
        )
        print(f"  Added GCMC-IAST: {len(df_iast_gcmc)} predictions")
    
    if all_iast_results_ml:
        df_iast_ml = pd.concat(all_iast_results_ml, ignore_index=True)
        df_result = df_result.merge(
            df_iast_ml,
            on=['MofName', 'Pressure[bar]', 'CO2Fraction'],
            how='left'
        )
        print(f"  Added ML-IAST: {len(df_iast_ml)} predictions")
    
    if all_iast_results_ml_pure:
        df_iast_ml_pure = pd.concat(all_iast_results_ml_pure, ignore_index=True)
        df_result = df_result.merge(
            df_iast_ml_pure,
            on=['MofName', 'Pressure[bar]', 'CO2Fraction'],
            how='left'
        )
        print(f"  Added MLPure-IAST: {len(df_iast_ml_pure)} predictions")
    
    return df_result


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_detailed_metrics(df_comparison):
    """Calculate detailed metrics for each MOF and composition (matching Notebook)."""
    metrics_list = []
    
    for mof in df_comparison['MofName'].unique():
        for frac in df_comparison['CO2Fraction'].unique():
            try:
                data = df_comparison[
                    (df_comparison['MofName'] == mof) & 
                    (df_comparison['CO2Fraction'] == frac)
                ].copy()
                
                if len(data) < 3:
                    continue
                
                # Prepare masks
                mask_co2 = data[['AdsCO2', 'AdsCO2GCMCIast', 'AdsCO2Predicted', 
                               'AdsCO2MLIast', 'AdsCO2MLPureIast']].notna().all(axis=1)
                mask_n2 = data[['AdsN2', 'AdsN2GCMCIast', 'AdsN2Predicted', 
                              'AdsN2MLIast', 'AdsN2MLPureIast']].notna().all(axis=1)
                
                result = {'MofName': mof, 'CO2Fraction': frac}
                
                # GCMC-IAST metrics
                if mask_co2.sum() > 0:
                    result['GCMC_IAST_CO2_R2'] = metrics.r2_score(
                        data.loc[mask_co2, 'AdsCO2'], data.loc[mask_co2, 'AdsCO2GCMCIast']
                    )
                    result['GCMC_IAST_CO2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_co2, 'AdsCO2'] - data.loc[mask_co2, 'AdsCO2GCMCIast']) / 
                        data.loc[mask_co2, 'AdsCO2']
                    ))
                
                if mask_n2.sum() > 0:
                    result['GCMC_IAST_N2_R2'] = metrics.r2_score(
                        data.loc[mask_n2, 'AdsN2'], data.loc[mask_n2, 'AdsN2GCMCIast']
                    )
                    result['GCMC_IAST_N2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_n2, 'AdsN2'] - data.loc[mask_n2, 'AdsN2GCMCIast']) / 
                        data.loc[mask_n2, 'AdsN2']
                    ))
                
                # ML direct prediction metrics
                if mask_co2.sum() > 0:
                    result['ML_CO2_R2'] = metrics.r2_score(
                        data.loc[mask_co2, 'AdsCO2'], data.loc[mask_co2, 'AdsCO2Predicted']
                    )
                    result['ML_CO2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_co2, 'AdsCO2'] - data.loc[mask_co2, 'AdsCO2Predicted']) / 
                        data.loc[mask_co2, 'AdsCO2']
                    ))
                
                if mask_n2.sum() > 0:
                    result['ML_N2_R2'] = metrics.r2_score(
                        data.loc[mask_n2, 'AdsN2'], data.loc[mask_n2, 'AdsN2Predicted']
                    )
                    result['ML_N2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_n2, 'AdsN2'] - data.loc[mask_n2, 'AdsN2Predicted']) / 
                        data.loc[mask_n2, 'AdsN2']
                    ))
                
                # ML-IAST metrics
                if mask_co2.sum() > 0:
                    result['ML_IAST_CO2_R2'] = metrics.r2_score(
                        data.loc[mask_co2, 'AdsCO2'], data.loc[mask_co2, 'AdsCO2MLIast']
                    )
                    result['ML_IAST_CO2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_co2, 'AdsCO2'] - data.loc[mask_co2, 'AdsCO2MLIast']) / 
                        data.loc[mask_co2, 'AdsCO2']
                    ))
                
                if mask_n2.sum() > 0:
                    result['ML_IAST_N2_R2'] = metrics.r2_score(
                        data.loc[mask_n2, 'AdsN2'], data.loc[mask_n2, 'AdsN2MLIast']
                    )
                    result['ML_IAST_N2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_n2, 'AdsN2'] - data.loc[mask_n2, 'AdsN2MLIast']) / 
                        data.loc[mask_n2, 'AdsN2']
                    ))
                
                # MLPure-IAST metrics
                if mask_co2.sum() > 0:
                    result['MLPure_IAST_CO2_R2'] = metrics.r2_score(
                        data.loc[mask_co2, 'AdsCO2'], data.loc[mask_co2, 'AdsCO2MLPureIast']
                    )
                    result['MLPure_IAST_CO2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_co2, 'AdsCO2'] - data.loc[mask_co2, 'AdsCO2MLPureIast']) / 
                        data.loc[mask_co2, 'AdsCO2']
                    ))
                
                if mask_n2.sum() > 0:
                    result['MLPure_IAST_N2_R2'] = metrics.r2_score(
                        data.loc[mask_n2, 'AdsN2'], data.loc[mask_n2, 'AdsN2MLPureIast']
                    )
                    result['MLPure_IAST_N2_MAPE'] = np.mean(np.abs(
                        (data.loc[mask_n2, 'AdsN2'] - data.loc[mask_n2, 'AdsN2MLPureIast']) / 
                        data.loc[mask_n2, 'AdsN2']
                    ))
                
                # Calculate selectivity MAPE (only for mixed compositions)
                if frac > 0.0 and frac < 1.0:
                    # Mask for valid selectivity calculation
                    mask_sel = data[['AdsCO2', 'AdsN2']].notna().all(axis=1) & (data['AdsN2'] > 1e-10)
                    
                    if mask_sel.sum() > 0:
                        # GCMC selectivity
                        s_gcmc = (data.loc[mask_sel, 'AdsCO2'] / data.loc[mask_sel, 'AdsN2']).values
                        
                        # ML selectivity and MAPE
                        if 'AdsCO2Predicted' in data.columns and 'AdsN2Predicted' in data.columns:
                            mask_ml = mask_sel & data[['AdsCO2Predicted', 'AdsN2Predicted']].notna().all(axis=1) & (data['AdsN2Predicted'] > 1e-10)
                            if mask_ml.sum() > 0:
                                s_ml = (data.loc[mask_ml, 'AdsCO2Predicted'] / data.loc[mask_ml, 'AdsN2Predicted']).values
                                s_gcmc_ml = (data.loc[mask_ml, 'AdsCO2'] / data.loc[mask_ml, 'AdsN2']).values
                                result['ML_S_MAPE'] = np.mean(np.abs((s_gcmc_ml - s_ml) / s_gcmc_ml))
                        
                        # GCMC-IAST selectivity and MAPE
                        if 'AdsCO2GCMCIast' in data.columns and 'AdsN2GCMCIast' in data.columns:
                            mask_gcmc_iast = mask_sel & data[['AdsCO2GCMCIast', 'AdsN2GCMCIast']].notna().all(axis=1) & (data['AdsN2GCMCIast'] > 1e-10)
                            if mask_gcmc_iast.sum() > 0:
                                s_gcmc_iast = (data.loc[mask_gcmc_iast, 'AdsCO2GCMCIast'] / data.loc[mask_gcmc_iast, 'AdsN2GCMCIast']).values
                                s_gcmc_ref = (data.loc[mask_gcmc_iast, 'AdsCO2'] / data.loc[mask_gcmc_iast, 'AdsN2']).values
                                result['GCMC_IAST_S_MAPE'] = np.mean(np.abs((s_gcmc_ref - s_gcmc_iast) / s_gcmc_ref))
                        
                        # ML-IAST selectivity and MAPE
                        if 'AdsCO2MLIast' in data.columns and 'AdsN2MLIast' in data.columns:
                            mask_ml_iast = mask_sel & data[['AdsCO2MLIast', 'AdsN2MLIast']].notna().all(axis=1) & (data['AdsN2MLIast'] > 1e-10)
                            if mask_ml_iast.sum() > 0:
                                s_ml_iast = (data.loc[mask_ml_iast, 'AdsCO2MLIast'] / data.loc[mask_ml_iast, 'AdsN2MLIast']).values
                                s_gcmc_ref = (data.loc[mask_ml_iast, 'AdsCO2'] / data.loc[mask_ml_iast, 'AdsN2']).values
                                result['ML_IAST_S_MAPE'] = np.mean(np.abs((s_gcmc_ref - s_ml_iast) / s_gcmc_ref))
                        
                        # MLPure-IAST selectivity and MAPE
                        if 'AdsCO2MLPureIast' in data.columns and 'AdsN2MLPureIast' in data.columns:
                            mask_mlp_iast = mask_sel & data[['AdsCO2MLPureIast', 'AdsN2MLPureIast']].notna().all(axis=1) & (data['AdsN2MLPureIast'] > 1e-10)
                            if mask_mlp_iast.sum() > 0:
                                s_mlp_iast = (data.loc[mask_mlp_iast, 'AdsCO2MLPureIast'] / data.loc[mask_mlp_iast, 'AdsN2MLPureIast']).values
                                s_gcmc_ref = (data.loc[mask_mlp_iast, 'AdsCO2'] / data.loc[mask_mlp_iast, 'AdsN2']).values
                                result['MLPure_IAST_S_MAPE'] = np.mean(np.abs((s_gcmc_ref - s_mlp_iast) / s_gcmc_ref))
                
                metrics_list.append(result)
                
            except Exception as e:
                continue
    
    return pd.DataFrame(metrics_list)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_comparison_for_mof(df_comparison, mof_name, co2_fraction, ax=None):
    """Plot comparison between GCMC, ML, and their respective IAST predictions (matching Notebook)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i / 9) for i in range(10)]
    
    data = df_comparison[(df_comparison['MofName'] == mof_name) & 
                        (df_comparison['CO2Fraction'] == co2_fraction)]
    
    if len(data) == 0:
        return ax
    
    # GCMC direct simulation
    ax.loglog(data['Pressure[bar]'], data['AdsCO2'], 'o', label='CO2 (GCMC)', color=colors[0], alpha=0.7)
    ax.loglog(data['Pressure[bar]'], data['AdsN2'], 's', label='N2 (GCMC)', color=colors[1], alpha=0.7)
    
    # GCMC-based IAST
    if 'AdsCO2GCMCIast' in data.columns:
        ax.loglog(data['Pressure[bar]'], data['AdsCO2GCMCIast'], '--', label='CO2 (GCMC-IAST)', color=colors[0])
        ax.loglog(data['Pressure[bar]'], data['AdsN2GCMCIast'], '--', label='N2 (GCMC-IAST)', color=colors[1])
    
    # ML predictions
    if 'AdsCO2Predicted' in data.columns:
        ax.loglog(data['Pressure[bar]'], data['AdsCO2Predicted'], '^', label='CO2 (ML)', color=colors[2], alpha=0.7)
        ax.loglog(data['Pressure[bar]'], data['AdsN2Predicted'], 'v', label='N2 (ML)', color=colors[3], alpha=0.7)
    
    # ML-based IAST
    if 'AdsCO2MLIast' in data.columns:
        ax.loglog(data['Pressure[bar]'], data['AdsCO2MLIast'], ':', label='CO2 (ML-IAST)', color=colors[2])
        ax.loglog(data['Pressure[bar]'], data['AdsN2MLIast'], ':', label='N2 (ML-IAST)', color=colors[3])
    
    # ML-Pure-IAST
    if 'AdsCO2MLPureIast' in data.columns:
        ax.loglog(data['Pressure[bar]'], data['AdsCO2MLPureIast'], '--', label='CO2 (ML-Pure-IAST)', color=colors[2])
        ax.loglog(data['Pressure[bar]'], data['AdsN2MLPureIast'], '--', label='N2 (ML-Pure-IAST)', color=colors[3])
    
    ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel('Adsorption (mmol/g)')
    ax.legend(fontsize=8)
    
    return ax


def create_comparison_grid(df_metrics, df_comparison, model_name, output_path, min_pressure=0.01, fractions=None):
    """Create grid of plots comparing GCMC, ML predictions and their IAST results (matching Notebook)."""
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    q_dic = {0: "$_{min}$", 1: "$_1$", 2: "$_2$", 3: "$_3$", 4: "$_{max}$"}
    
    fig = plt.figure(figsize=(25, 5*len(fractions)))
    gs = gridspec.GridSpec(len(fractions), len(quantiles), figure=fig, wspace=0.3, hspace=0.3)
    
    df_filtered = df_comparison[df_comparison['Pressure[bar]'] >= min_pressure]
    
    for row, co2_fraction in enumerate(fractions):
        # Get performance data for current composition
        df_frac = df_metrics[df_metrics['CO2Fraction'] == co2_fraction].copy()
        
        if len(df_frac) == 0:
            continue
        
        # Find representative MOFs based on ML MAPE
        representative_mofs = []
        for q in quantiles:
            if 'ML_CO2_MAPE' in df_frac.columns:
                target_mape = df_frac['ML_CO2_MAPE'].quantile(q)
                closest_idx = (df_frac['ML_CO2_MAPE'] - target_mape).abs().idxmin()
                representative_mofs.append(df_frac.loc[closest_idx])
            else:
                break
        
        if len(representative_mofs) == 0:
            continue
        
        for col, mof_data in enumerate(representative_mofs):
            ax = fig.add_subplot(gs[row, col])
            
            # Plot comparison
            plot_comparison_for_mof(df_filtered, mof_data['MofName'], co2_fraction, ax)
            
            # Add performance metrics text
            if co2_fraction == 0.0:
                metrics_text = (
                    f"MAPP(N$_2$): R$^2$={mof_data.get('ML_N2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_N2_MAPE', np.nan):.3f}\n"
                    f"MAPP-IAST(N$_2$): R$^2$={mof_data.get('ML_IAST_N2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_IAST_N2_MAPE', np.nan):.3f}"
                )
            elif co2_fraction == 1.0:
                metrics_text = (
                    f"MAPP(CO$_2$): R$^2$={mof_data.get('ML_CO2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_CO2_MAPE', np.nan):.3f}\n"
                    f"MAPP-IAST(CO$_2$): R$^2$={mof_data.get('ML_IAST_CO2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_IAST_CO2_MAPE', np.nan):.3f}"
                )
            else:
                metrics_text = (
                    f"MAPP(CO$_2$): R$^2$={mof_data.get('ML_CO2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_CO2_MAPE', np.nan):.3f}\n"
                    f"MAPP-IAST(CO$_2$): R$^2$={mof_data.get('ML_IAST_CO2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_IAST_CO2_MAPE', np.nan):.3f}\n"
                    f"MAPP(N$_2$): R$^2$={mof_data.get('ML_N2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_N2_MAPE', np.nan):.3f}\n"
                    f"MAPP-IAST(N$_2$): R$^2$={mof_data.get('ML_IAST_N2_R2', np.nan):.3f}, MAPE={mof_data.get('ML_IAST_N2_MAPE', np.nan):.3f}"
                )
            
            ax.text(0.95, 0.05, metrics_text,
                   transform=ax.transAxes,
                   ha='right', va='bottom',
                   fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.5))
            
            # Set titles and labels
            if row == 0:
                ax.set_title(f"MAPE-Q{q_dic[col]}\nMOF: {mof_data['MofName']}", fontsize=12, fontweight='bold')
            else:
                ax.set_title(f"MOF: {mof_data['MofName']}", fontsize=12)
            
            if col == 0:
                ax.text(-0.2, 0.5, f'CO$_2$/N$_2$ = {co2_fraction*100:.0f}/{(1-co2_fraction)*100:.0f}',
                        transform=ax.transAxes,
                        rotation=90,
                        verticalalignment='center',
                        fontsize=14,
                        fontweight='bold')
            
            # Only show legend for the first plot
            if col == 0 and row == 0:
                ax.legend(loc='upper left', fontsize=7)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed comparison grid to: {output_path}")


def plot_mape_boxplots_by_fraction(df_metrics, output_path):
    """Generate box plots comparing MAPE across methods for different CO2 fractions (matching Notebook)."""
    
    def prepare_data_for_box(metric_col):
        """Prepare data for box plot visualization."""
        data_list = []
        
        # Define method mapping
        method_configs = [
            ('ML', 'MAPP'),
            ('ML_IAST', 'MAPP+IAST'),
            ('MLPure_IAST', 'MAPPPure+IAST'),
            ('GCMC_IAST', 'GCMC+IAST')
        ]
        
        for prefix, method_name in method_configs:
            col_name = f"{prefix}_{metric_col}"
            if col_name in df_metrics.columns:
                temp_df = df_metrics[['CO2Fraction', col_name]].copy()
                temp_df['Method'] = method_name
                temp_df.rename(columns={col_name: metric_col}, inplace=True)
                data_list.append(temp_df)
        
        if len(data_list) == 0:
            return None
        
        return pd.concat(data_list, ignore_index=True)
    
    # Prepare data for each metric
    metrics = [
        ('CO2_MAPE', r'$\mathbf{Q_{CO_2}}$ (mol/kg)'),
        ('N2_MAPE', r'$\mathbf{Q_{N_2}}$ (mol/kg)')
    ]
    
    # Add selectivity metric
    metrics.append(('S_MAPE', r'$\mathbf{S_{CO_2/N_2}}$'))
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Color scheme
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i / 5) for i in range(4)]
    
    # Plot each metric
    for idx, (metric_name, metric_label) in enumerate(metrics):
        data = prepare_data_for_box(metric_name)
        
        if data is None or len(data) == 0:
            continue
        
        # Draw box plot
        sns.boxplot(data=data,
                   x='CO2Fraction',
                   y=metric_name,
                   hue='Method',
                   ax=axes[idx],
                   palette=colors,
                   fliersize=2)
        
        # Set labels and title
        axes[idx].set_title(f'Performance of {metric_label} Prediction', 
                           fontsize=15, pad=10, fontweight='bold')
        axes[idx].set_xlabel('CO$_2$ Fraction', fontsize=14)
        axes[idx].set_ylabel('MAPE', fontsize=14)
        
        # Set tick label size
        axes[idx].tick_params(axis='both', labelsize=12)
        
        # Adjust legend
        axes[idx].legend(loc='upper right',
                        borderaxespad=0.2,
                        framealpha=0.5,
                        fontsize=10)
        
        # Set y-axis range based on percentile
        if metric_name == 'CO2_MAPE':
            y_max = np.percentile(data[metric_name].dropna(), 99.9)
        else:
            y_max = np.percentile(data[metric_name].dropna(), 95)
        
        axes[idx].set_ylim(0, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MAPE box plots to: {output_path}")


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function for comprehensive MAPP vs IAST comparison."""
    # Suppress all warnings from pyGAPS and numerical calculations
    warnings.filterwarnings('ignore', category=UserWarning, module='pygaps')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pygaps')
    warnings.filterwarnings('ignore', message='.*extrapolate.*')
    warnings.filterwarnings('ignore', message='.*highest pressure.*')
    warnings.filterwarnings('ignore', message='.*invalid value.*')
    warnings.filterwarnings('ignore', message='.*log.*')
    
    results_dir = PROJECT_ROOT / config["output"]["results"]
    results_dir.mkdir(exist_ok=True, parents=True)
    
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("Comprehensive MAPP and IAST Methods Comparison")
    print("=" * 70)
    
    # Load GCMC data
    df_gcmc = load_gcmc_data()
    
    # Load MAPP mixture model predictions
    inference_dir = PROJECT_ROOT / "MOFTransformer" / "inference"
    dataset = "ddmof"
    tasks = ["SymlogAbsLoadingCO2", "SymlogAbsLoadingN2"]
    
    try:
        df_pred_mix = load_ml_predictions(inference_dir, args.model_name, tasks, dataset)
        
        # Set pure component predictions to NaN (mixture model not trained on pure)
        df_pred_mix.loc[df_pred_mix["CO2Fraction"] == 0, "AdsCO2Predicted"] = None
        df_pred_mix.loc[df_pred_mix["CO2Fraction"] == 1, "AdsN2Predicted"] = None
        df_pred_mix.loc[df_pred_mix["CO2Fraction"] == 0, "AdsCO2PredictedStd"] = None
        df_pred_mix.loc[df_pred_mix["CO2Fraction"] == 1, "AdsN2PredictedStd"] = None
        
    except FileNotFoundError as e:
        print(f"Error loading mixture model: {e}")
        return
    
    # Load MAPP pure model predictions
    try:
        df_pred_pure = load_ml_predictions(inference_dir, args.pure_model_name, tasks, dataset)
        
        # Rename columns to indicate pure model
        pred_cols = [c for c in df_pred_pure.columns if c.startswith("Ads") and "Predicted" in c]
        df_pred_pure = df_pred_pure.rename(
            columns={c: c.replace("Predicted", "PredictedPure") for c in pred_cols}
        )
        
        # Drop uncertainty columns to avoid conflicts
        drop_cols = [c for c in df_pred_pure.columns if "Uncertainty" in c or "Std" in c]
        df_pred_pure = df_pred_pure.drop(columns=drop_cols)
        
    except FileNotFoundError as e:
        print(f"Warning: Pure model not found: {e}")
        df_pred_pure = None
    
    # Merge all data
    df_compare = df_gcmc.merge(
        df_pred_mix[["MofName", "CO2Fraction", "Pressure[bar]", "AdsCO2Predicted", "AdsN2Predicted"]],
        on=["MofName", "CO2Fraction", "Pressure[bar]"],
        how="inner"
    )
    
    if df_pred_pure is not None:
        df_compare = df_compare.merge(
            df_pred_pure,
            on=["MofName", "CO2Fraction", "Pressure[bar]"],
            how="outer"
        )
    
    print(f"Merged data: {len(df_compare)} samples, {df_compare['MofName'].nunique()} MOFs")
    
    # Filter low pressures
    df_compare = df_compare[df_compare["Pressure[bar]"] >= 0.01]
    print(f"Filtered (P>=0.01 bar): {len(df_compare)} samples")
    
    # Calculate all IAST predictions
    if HAS_PYGAPS:
        df_comparison = calculate_all_iast(df_compare)
    else:
        df_comparison = df_compare
        print("\nSkipping IAST calculations (pyGAPS not available)")
    
    # Save comprehensive comparison data
    output_file = results_dir / f"iast_comparison_all_{args.model_name}.csv"
    df_comparison.to_csv(output_file, index=False)
    print(f"\nSaved comparison data to: {output_file}")
    
    # Calculate detailed metrics
    print("\nCalculating detailed metrics...")
    df_metrics = calculate_detailed_metrics(df_comparison)
    
    if len(df_metrics) > 0:
        metrics_file = results_dir / f"iast_metrics_detailed_{args.model_name}.csv"
        df_metrics.to_csv(metrics_file, index=False, float_format='%.6f')
        print(f"Saved detailed metrics to: {metrics_file}")
        
        # Generate visualizations
        print("\nGenerating detailed comparison grid...")
        plot_path = fig_dir / f"iast_methods_comparison_{args.model_name}.png"
        create_comparison_grid(df_metrics, df_comparison, args.model_name, plot_path)
        
        print("\nGenerating MAPE box plots by CO2 fraction...")
        boxplot_path = fig_dir / f"iast_methods_mape_boxplots_{args.model_name}.png"
        plot_mape_boxplots_by_fraction(df_metrics, boxplot_path)
    
    print("\nComprehensive MAPP vs IAST comparison complete.")


if __name__ == "__main__":
    # Extract default model names from config
    default_mixture_model = config["models"]["MAPP_GMOF"]
    default_pure_model = config["models"]["MAPPPure"]
    
    # Extract model name from full path (last part after '/')
    default_mixture_name = default_mixture_model.split('/')[-2] + "_" + default_mixture_model.split('/')[-1]
    default_pure_name = default_pure_model.split('/')[-2] + "_" + default_pure_model.split('/')[-1]
    
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of MAPP and IAST methods"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default=default_mixture_name,
        help=f"MAPP mixture model name (default from config: {default_mixture_name})"
    )
    parser.add_argument(
        "--pure-model-name",
        type=str,
        default=default_pure_name,
        help=f"MAPP pure component model name (default from config: {default_pure_name})"
    )
    
    args = parser.parse_args()
    main(args)
