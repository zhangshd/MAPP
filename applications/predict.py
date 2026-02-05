#!/usr/bin/env python3
"""
MAPP Prediction Application

User-friendly script for predicting CO2/N2 adsorption, working capacity,
and selectivity from CIF files using MAPP model.

Usage:
    conda run -n mofnn python applications/predict.py --cif_dir PATH [options]

Example:
    python applications/predict.py --cif_dir applications/demo_cifs
    python applications/predict.py --cif_dir applications/demo_cifs --use_iast

Author: zhangshd
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import logging

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "MOFTransformer"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = PROJECT_ROOT / "experiments" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Default PSA-like conditions (flue gas)
DEFAULT_ADS_PRESSURE = 1.0      # bar
DEFAULT_ADS_CO2_FRAC = 0.15     # 15% CO2
DEFAULT_DES_PRESSURE = 0.01    # bar (VSA)
DEFAULT_DES_CO2_FRAC = 0.9      # 90% CO2

# Symlog transform parameters
SYMLOG_THRESHOLD = 1e-4


def symlog_inverse(y: np.ndarray, threshold: float = SYMLOG_THRESHOLD) -> np.ndarray:
    """Inverse symlog transform: y -> x = sign(y) * threshold * (10^|y| - 1)"""
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def calculate_selectivity(q_co2: float, q_n2: float, x_co2: float) -> float:
    """
    Calculate CO2/N2 selectivity.
    S = (y_CO2 / y_N2) / (x_CO2 / x_N2) = (q_CO2 * x_N2) / (q_N2 * x_CO2)
    """
    x_n2 = 1 - x_co2
    if q_n2 < 1e-10 or x_co2 < 1e-10 or x_n2 < 1e-10:
        return np.nan
    selectivity = (q_co2 * x_n2) / (q_n2 * x_co2)
    if selectivity > 1e6 or selectivity < 1e-6:
        return np.nan
    return selectivity


def run_mapp_inference(
    cif_list: list,
    pressures: list,
    co2_fracs: list,
    model_dir: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Run MAPP inference for given conditions.
    
    Returns DataFrame with columns: CifId, Pressure[bar], CO2Fraction, 
                                    Nco2, Nn2, Nco2Uncertainty, Nn2Uncertainty
    """
    from inference import inference
    
    # Build inputs list: [cif_id, pressure, co2frac]
    inputs = []
    for cif in cif_list:
        cif_id = Path(cif).stem
        for p in pressures:
            for frac in co2_fracs:
                inputs.append([cif_id, p, frac])
    
    logger.info(f"Running MAPP inference: {len(cif_list)} MOFs × {len(pressures)} pressures × {len(co2_fracs)} fracs = {len(inputs)} predictions")
    
    # Check for uncertainty trees
    uncertainty_trees_file = model_dir / "uncertainty_trees.pkl"
    if not uncertainty_trees_file.exists():
        uncertainty_trees_file = None
    
    # Run inference (save_csv=False to avoid redundant output files)
    outputs = inference(
        cif_list=cif_list,
        model_dir=str(model_dir),
        saved_dir=output_dir,
        inputs=inputs,
        clean=True,
        repeat=1,
        uncertainty_trees_file=uncertainty_trees_file,
        save_csv=False
    )
    
    # Combine task outputs into single DataFrame
    dfs = []
    for task_name, task_data in outputs.items():
        # Determine gas name from task name
        if 'CO2' in task_name.upper():
            gas = 'co2'
        elif 'N2' in task_name.upper():
            gas = 'n2'
        else:
            gas = task_name.lower()
        
        df_data = {
            'CifId': task_data['CifId'],
            'Pressure[bar]': task_data['Pressure[bar]'],
            'CO2Fraction': task_data['CO2Fraction'],
            f'N{gas}Symlog': task_data['Predicted'],
        }
        
        # Add uncertainty if available
        if 'Uncertainty' in task_data and task_data['Uncertainty'] is not None:
            df_data[f'N{gas}Uncertainty'] = task_data['Uncertainty']
        
        dfs.append(pd.DataFrame(df_data))
    
    # Merge all tasks
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=['CifId', 'Pressure[bar]', 'CO2Fraction'])
    
    return result


def process_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply symlog inverse transform and rename columns."""
    # Apply symlog inverse to Symlog columns
    symlog_cols = [c for c in df.columns if 'Symlog' in c]
    for col in symlog_cols:
        # Create raw column name: Nco2Symlog -> Nco2[mol/kg]
        raw_col = col.replace('Symlog', '[mol/kg]')
        df[raw_col] = symlog_inverse(df[col].values)
    
    # Drop Symlog columns
    df = df.drop(columns=symlog_cols)
    
    # Reorder columns: put Uncertainty columns at the end
    uncertainty_cols = [c for c in df.columns if 'Uncertainty' in c]
    other_cols = [c for c in df.columns if 'Uncertainty' not in c]
    df = df[other_cols + uncertainty_cols]
    
    return df


def compute_metrics(
    df: pd.DataFrame,
    ads_pressure: float,
    ads_co2_frac: float,
    des_pressure: float,
    des_co2_frac: float,
    suffix: str = ''
) -> pd.DataFrame:
    """
    Compute working capacity and selectivity from raw predictions.
    
    Returns DataFrame with columns: MofName, WCco2, Sco2Ads, etc.
    """
    # Filter conditions
    def filter_condition(df, p, frac, tol=0.01):
        return df[
            (np.abs(df['Pressure[bar]'] - p) < tol) &
            (np.abs(df['CO2Fraction'] - frac) < tol)
        ].copy()
    
    ads_df = filter_condition(df, ads_pressure, ads_co2_frac).drop_duplicates(subset=['CifId'])
    des_df = filter_condition(df, des_pressure, des_co2_frac).drop_duplicates(subset=['CifId'])
    
    if len(ads_df) == 0 or len(des_df) == 0:
        logger.warning(f"No data found for specified conditions{suffix}")
        return pd.DataFrame()
    
    # Detect CO2 and N2 column names (format: Nco2[mol/kg], Nn2[mol/kg])
    co2_col = next((c for c in df.columns if 'co2' in c.lower() and 'mol/kg' in c), None)
    n2_col = next((c for c in df.columns if 'n2' in c.lower() and 'mol/kg' in c), None)
    
    if not co2_col or not n2_col:
        logger.error(f"Cannot find CO2/N2 columns in: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Merge ads and des data
    merged = ads_df[['CifId', co2_col, n2_col]].merge(
        des_df[['CifId', co2_col, n2_col]],
        on='CifId',
        suffixes=('Ads', 'Des')
    )
    
    results = []
    for _, row in merged.iterrows():
        q_co2_ads = row[f'{co2_col}Ads']
        q_n2_ads = row[f'{n2_col}Ads']
        q_co2_des = row[f'{co2_col}Des']
        q_n2_des = row[f'{n2_col}Des']
        
        wc_co2 = q_co2_ads - q_co2_des
        wc_n2 = q_n2_ads - q_n2_des
        s_ads = calculate_selectivity(q_co2_ads, q_n2_ads, ads_co2_frac)
        s_des = calculate_selectivity(q_co2_des, q_n2_des, des_co2_frac)
        
        results.append({
            'MofName': row['CifId'],
            f'WCco2{suffix}': wc_co2,
            f'WCn2{suffix}': wc_n2,
            f'Sco2Ads{suffix}': s_ads,
            f'Sco2Des{suffix}': s_des,
            f'Nco2Ads{suffix}[mol/kg]': q_co2_ads,
            f'Nn2Ads{suffix}[mol/kg]': q_n2_ads,
            f'Nco2Des{suffix}[mol/kg]': q_co2_des,
            f'Nn2Des{suffix}[mol/kg]': q_n2_des,
        })
    
    return pd.DataFrame(results)


def run_iast_workflow(
    df_raw: pd.DataFrame,
    mof_names: list,
    ads_pressure: float,
    ads_co2_frac: float,
    des_pressure: float,
    des_co2_frac: float,
    output_dir: Path
) -> pd.DataFrame:
    """
    Run IAST workflow: fit isotherms and predict mixture adsorption.
    
    Args:
        df_raw: DataFrame with pure component predictions (CO2Fraction=0 or 1)
        
    Returns:
        DataFrame with IAST metrics
    """
    from utils_iast import fit_isotherm, iast_predict, save_models_json
    
    # Detect column names (format: Nco2[mol/kg], Nn2[mol/kg])
    co2_col = next((c for c in df_raw.columns if 'co2' in c.lower() and 'mol/kg' in c), None)
    n2_col = next((c for c in df_raw.columns if 'n2' in c.lower() and 'mol/kg' in c), None)
    
    # Filter pure component data
    df_co2 = df_raw[np.abs(df_raw['CO2Fraction'] - 1.0) < 0.01].copy()
    df_n2 = df_raw[np.abs(df_raw['CO2Fraction'] - 0.0) < 0.01].copy()
    
    if len(df_co2) == 0 or len(df_n2) == 0:
        logger.error("No pure component data found for IAST")
        return pd.DataFrame()
    
    # Fit isotherms for each MOF
    all_models = {}
    iast_results = []
    
    for mof_name in mof_names:
        mof_co2 = df_co2[df_co2['CifId'] == mof_name].sort_values('Pressure[bar]')
        mof_n2 = df_n2[df_n2['CifId'] == mof_name].sort_values('Pressure[bar]')
        
        if len(mof_co2) < 3 or len(mof_n2) < 3:
            logger.warning(f"Insufficient pure data for {mof_name}, skipping IAST")
            continue
        
        # Fit isotherms
        model_co2 = fit_isotherm(
            mof_co2['Pressure[bar]'].values,
            mof_co2[co2_col].values,
            mof_name, 'CO2'
        )
        model_n2 = fit_isotherm(
            mof_n2['Pressure[bar]'].values,
            mof_n2[n2_col].values,
            mof_name, 'N2'
        )
        
        if not model_co2 or not model_n2:
            logger.warning(f"Isotherm fitting failed for {mof_name}")
            continue
        
        all_models[mof_name] = {'CO2': model_co2, 'N2': model_n2}
        
        # IAST predictions
        q_co2_ads_iast, q_n2_ads_iast = iast_predict(model_co2, model_n2, ads_pressure, ads_co2_frac)
        q_co2_des_iast, q_n2_des_iast = iast_predict(model_co2, model_n2, des_pressure, des_co2_frac)
        
        wc_co2_iast = q_co2_ads_iast - q_co2_des_iast
        wc_n2_iast = q_n2_ads_iast - q_n2_des_iast
        s_ads_iast = calculate_selectivity(q_co2_ads_iast, q_n2_ads_iast, ads_co2_frac)
        s_des_iast = calculate_selectivity(q_co2_des_iast, q_n2_des_iast, des_co2_frac)
        
        iast_results.append({
            'MofName': mof_name,
            'WCco2IAST': wc_co2_iast,
            'WCn2IAST': wc_n2_iast,
            'Sco2AdsIAST': s_ads_iast,
            'Sco2DesIAST': s_des_iast,
            'Nco2AdsIAST[mol/kg]': q_co2_ads_iast,
            'Nn2AdsIAST[mol/kg]': q_n2_ads_iast,
            'Nco2DesIAST[mol/kg]': q_co2_des_iast,
            'Nn2DesIAST[mol/kg]': q_n2_des_iast,
        })
    
    # Save models
    if all_models:
        save_models_json(all_models, output_dir / 'isotherm_models.json')
    
    return pd.DataFrame(iast_results)


def main(args):
    """Main prediction workflow."""
    print("=" * 70)
    print("MAPP Adsorption Prediction")
    print("=" * 70)
    
    # Setup paths
    cif_path = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = PROJECT_ROOT / CONFIG['models']['MAPP_GMOF']
    
    # Get CIF files
    if cif_path.is_file():
        cif_list = [cif_path]
    else:
        cif_list = list(cif_path.glob("*.cif"))
    
    if not cif_list:
        logger.error(f"No CIF files found in {cif_path}")
        return
    
    print(f"\nInput: {len(cif_list)} CIF files")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"\nConditions:")
    print(f"  Adsorption: {args.ads_pressure} bar, CO2 = {args.ads_co2_frac}")
    print(f"  Desorption: {args.des_pressure} bar, CO2 = {args.des_co2_frac}")
    
    # Define prediction conditions
    pressures = [args.ads_pressure, args.des_pressure]
    co2_fracs = [args.ads_co2_frac, args.des_co2_frac]
    
    if args.use_iast:
        # Add pure component conditions for IAST
        pure_pressures = np.logspace(-2, 1, 22).tolist()  # 0.01 to 10 bar
        pressures = sorted(set(pressures + pure_pressures))
        co2_fracs = sorted(set(co2_fracs + [0.0, 1.0]))
        print(f"\nIAST Mode: Adding {len(pure_pressures)} pressure points for pure components")
    
    # Run MAPP inference
    print("\n[1/3] Running MAPP inference...")
    df_pred = run_mapp_inference(cif_list, pressures, co2_fracs, model_dir, output_dir)
    
    # Apply symlog inverse
    print("[2/3] Processing predictions...")
    df_raw = process_predictions(df_pred)
    
    # Save raw predictions
    raw_file = output_dir / 'mapp_raw_predictions.csv'
    df_raw.to_csv(raw_file, index=False, float_format='%.6f')
    print(f"  Saved: {raw_file}")
    
    # Compute metrics
    print("[3/3] Computing performance metrics...")
    df_metrics = compute_metrics(
        df_raw, args.ads_pressure, args.ads_co2_frac,
        args.des_pressure, args.des_co2_frac
    )
    
    # IAST workflow
    if args.use_iast and len(df_metrics) > 0:
        print("\n[IAST] Running IAST workflow...")
        mof_names = df_metrics['MofName'].drop_duplicates().tolist()
        df_iast = run_iast_workflow(
            df_raw, mof_names,
            args.ads_pressure, args.ads_co2_frac,
            args.des_pressure, args.des_co2_frac,
            output_dir
        )
        if len(df_iast) > 0:
            df_metrics = df_metrics.merge(df_iast, on='MofName', how='left')
    
    # Reorder columns: WC and S first
    if len(df_metrics) > 0:
        priority_cols = ['MofName', 'WCco2', 'WCn2', 'Sco2Ads', 'Sco2Des']
        if args.use_iast:
            priority_cols += ['WCco2IAST', 'WCn2IAST', 'Sco2AdsIAST', 'Sco2DesIAST']
        other_cols = [c for c in df_metrics.columns if c not in priority_cols]
        df_metrics = df_metrics[priority_cols + other_cols]
        
        summary_file = output_dir / 'mapp_summary.csv'
        df_metrics.to_csv(summary_file, index=False, float_format='%.6f')
        print(f"  Saved: {summary_file}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    if len(df_metrics) > 0:
        print("\nSummary Preview:")
        print(df_metrics[['MofName', 'WCco2', 'Sco2Ads']].head().to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MAPP Adsorption Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cif_dir', type=str, required=True,
                        help='Directory containing CIF files or a single CIF file')
    parser.add_argument('--ads_pressure', type=float, default=DEFAULT_ADS_PRESSURE,
                        help='Adsorption pressure (bar)')
    parser.add_argument('--ads_co2_frac', type=float, default=DEFAULT_ADS_CO2_FRAC,
                        help='Adsorption CO2 fraction')
    parser.add_argument('--des_pressure', type=float, default=DEFAULT_DES_PRESSURE,
                        help='Desorption pressure (bar)')
    parser.add_argument('--des_co2_frac', type=float, default=DEFAULT_DES_CO2_FRAC,
                        help='Desorption CO2 fraction')
    parser.add_argument('--use_iast', action='store_true',
                        help='Enable IAST comparison')
    parser.add_argument('--output_dir', type=str, 
                        default=str(PROJECT_ROOT / 'applications' / 'output'),
                        help='Output directory')
    
    args = parser.parse_args()
    main(args)
