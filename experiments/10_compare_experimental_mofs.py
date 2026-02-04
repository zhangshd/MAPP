#!/usr/bin/env python3
"""
Experimental MOF Validation Script

Compare ML predictions with experimental data and RASPA2 simulations for selected MOFs.
Analyzes:
1. Pure component isotherms (CO2, N2) vs experimental and GCMC data
2. Flue gas capture performance using IAST calculations
3. Working capacity and selectivity for PSA/VSA conditions

Usage:
    conda run -n mofnn python 10_compare_experimental_mofs.py [--model-name NAME]
    
Environment:
    Requires mofnn conda environment with pygaps, pandas, numpy, matplotlib, seaborn
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Flue gas PSA conditions
FLUE_GAS_CONDITIONS = {
    'adsorption': {
        'pressure': 1.0,  # bar
        'co2_fraction': 0.15,
        'n2_fraction': 0.85
    },
    'desorption': {
        'pressure': 0.01,  # bar
        'co2_fraction': 0.9,
        'n2_fraction': 0.1
    }
}

# MOF experimental data directories
MOF_DIRS = {
    'CALF20': 'RASPATOOLS/examples/raspa2_validation/CALF20',
    'IRMOF1': 'RASPATOOLS/examples/raspa2_validation/IRMOF1',
    'UiO66': 'RASPATOOLS/examples/raspa2_validation/UiO66',
}


# =============================================================================
# Utility Functions
# =============================================================================

def symlog(x, threshold=1e-4):
    """Symmetric log transform: sign(x) * log10(1 + |x|/threshold)"""
    return np.sign(x) * np.log10(1 + np.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog transform: sign(y) * threshold * (10^|y| - 1)"""
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def map_composition(df):
    """Map gas composition to CO2 fraction"""
    if df["GasName"] == "CO2":
        return round(df["MoleFraction"], 4)
    else:
        return round((1 - df["MoleFraction"]), 4)


def filter_by_closest_temperature(df, target_temp=298, temp_tolerance=10):
    """Filter data to closest target temperature
    
    Args:
        df: DataFrame with Temperature[K] column
        target_temp: Target temperature in K (default 298)
        temp_tolerance: Maximum allowed temperature deviation in K
    
    Returns:
        Filtered DataFrame or None if no data within tolerance
    """
    if df is None or len(df) == 0:
        return df
    if 'Temperature[K]' not in df.columns:
        return df
    temps = df['Temperature[K]'].unique()
    closest_temp = min(temps, key=lambda x: abs(x - target_temp))
    if abs(closest_temp - target_temp) > temp_tolerance:
        return None
    return df[df['Temperature[K]'] == closest_temp].copy()


def calculate_metrics(y_true, y_pred, label=""):
    """Calculate MAE, RMSE and R² metrics with NaN safety"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if mask.sum() < 2:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'N': 0}
    
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    r2 = r2_score(y_true[mask], y_pred[mask])
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'N': mask.sum()}


# =============================================================================
# IAST Calculation Functions
# =============================================================================

def fit_pure_isotherms_from_data(pressures, loadings, gas_name, mof_name, temperature=298):
    """Fit pure component isotherm using pygaps
    
    Args:
        pressures: Array of pressures in bar
        loadings: Array of loadings in mol/kg
        gas_name: 'CO2' or 'N2'
        mof_name: Name of the MOF
        temperature: Temperature in K
    
    Returns:
        ModelIsotherm object or None if fitting fails
    """
    if not HAS_PYGAPS or len(pressures) < 3:
        return None
    
    # Create DataFrame for isotherm
    data = pd.DataFrame({
        'Pressure[bar]': pressures,
        'AbsLoading': loadings * 1000  # Convert mol/kg to mmol/g
    })
    
    try:
        # Create isotherm object
        isotherm = pg.PointIsotherm(
            isotherm_data=data,
            pressure_key='Pressure[bar]',
            loading_key='AbsLoading',
            material=mof_name,
            adsorbate=gas_name,
            temperature=temperature - 273.15,  # Convert K to °C
            pressure_mode='absolute',
            pressure_unit='bar',
            loading_basis='molar',
            loading_unit='mmol',
            material_basis='mass',
            material_unit='g',
            temperature_unit='°C'
        )
        
        # Fit model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_iso = pgm.model_iso(
                isotherm, 
                model=["Henry", "Langmuir", "DSLangmuir"],
                verbose=False
            )
        return model_iso
    except Exception as e:
        # print(f"Error fitting {gas_name} isotherm for {mof_name}: {e}")
        return None


def calculate_iast_loading(iso_co2, iso_n2, total_pressure, co2_fraction):
    """Calculate mixture loading using IAST
    
    Args:
        iso_co2: ModelIsotherm for CO2
        iso_n2: ModelIsotherm for N2
        total_pressure: Total pressure in bar
        co2_fraction: Mole fraction of CO2
    
    Returns:
        Tuple of (co2_loading, n2_loading) in mol/kg or (None, None) if calculation fails
    """
    if iso_co2 is None or iso_n2 is None:
        return None, None
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mix_ads = pgi.iast_point_fraction(
                [iso_co2, iso_n2],
                [co2_fraction, 1 - co2_fraction],
                total_pressure,
                verbose=False
            )
        # Convert from mmol/g to mol/kg
        return mix_ads[0] / 1000, mix_ads[1] / 1000
    except Exception:
        return None, None


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_prediction_data(mixture_model_name):
    """Load ML prediction data for CO2 and N2"""
    inference_dir = PROJECT_ROOT / "MOFTransformer" / "inference" / "exp_MOF"
    
    co2_pred_file = inference_dir / f"SymlogAbsLoadingCO2_predictions_{mixture_model_name}.csv"
    n2_pred_file = inference_dir / f"SymlogAbsLoadingN2_predictions_{mixture_model_name}.csv"
    
    if not co2_pred_file.exists() or not n2_pred_file.exists():
        raise FileNotFoundError(f"Prediction files not found in {inference_dir}")
    
    df_co2 = pd.read_csv(co2_pred_file)
    df_n2 = pd.read_csv(n2_pred_file)
    
    # Apply inverse symlog transform
    df_co2["PredictedInit"] = symlog_inverse(df_co2["Predicted"])
    df_n2["PredictedInit"] = symlog_inverse(df_n2["Predicted"])
    
    print(f"Loaded predictions:")
    print(f"  CO2: {len(df_co2)} rows")
    print(f"  N2: {len(df_n2)} rows")
    
    return df_co2, df_n2


def load_experimental_data():
    """Load experimental and simulation data for each MOF"""
    mof_data = {}
    
    for mof_name, mof_rel_dir in MOF_DIRS.items():
        mof_dir = PROJECT_ROOT / mof_rel_dir
        try:
            sim_file = mof_dir / 'simulation_data.csv'
            exp_co2_file = mof_dir / 'experimental_data_CO2.csv'
            exp_n2_file = mof_dir / 'experimental_data_N2.csv'
            
            data = {}
            if sim_file.exists():
                data['sim'] = pd.read_csv(sim_file)
                data['sim']["CO2Fraction"] = data['sim'].apply(map_composition, axis=1)
            if exp_co2_file.exists():
                data['exp_CO2'] = pd.read_csv(exp_co2_file)
            if exp_n2_file.exists():
                data['exp_N2'] = pd.read_csv(exp_n2_file)
            
            mof_data[mof_name] = data
            print(f"Loaded data for {mof_name}")
        except Exception as e:
            print(f"Error loading {mof_name}: {e}")
    
    print(f"\nTotal MOFs loaded: {len(mof_data)}")
    return mof_data


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_pure_component_comparison(df_co2, df_n2, mof_data, output_dir):
    """Generate comparison plots for pure component isotherms"""
    # Prepare pure component data
    df_co2_pure = df_co2[df_co2['CO2Fraction'] == 1.0].copy()
    df_n2_pure = df_n2[df_n2['CO2Fraction'] == 0.0].copy()
    
    # Color scheme
    cmap = plt.get_cmap("rainbow")
    n_colors = 6
    base_colors = [cmap(i / n_colors) for i in range(n_colors)]
    colors = {
        'CO2': {'exp': base_colors[0], 'sim': base_colors[1], 'pred': base_colors[2]},
        'N2': {'exp': base_colors[3], 'sim': base_colors[4], 'pred': base_colors[5]}
    }
    
    available_mofs = [m for m in mof_data.keys() if m in df_co2_pure['CifId'].values]
    n_mofs = len(available_mofs)
    n_cols = 3
    n_rows = (n_mofs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    p_min, p_max = 0, 10
    target_temp = 298
    
    for idx, mof_name in enumerate(available_mofs):
        ax = axes[idx]
        data = mof_data[mof_name]
        
        # Get prediction data
        co2_pred = df_co2_pure[df_co2_pure['CifId'] == mof_name].sort_values('Pressure[bar]')
        n2_pred = df_n2_pure[df_n2_pure['CifId'] == mof_name].sort_values('Pressure[bar]')
        
        # Get experimental and simulation data
        exp_co2 = filter_by_closest_temperature(data.get('exp_CO2'), target_temp)
        exp_n2 = filter_by_closest_temperature(data.get('exp_N2'), target_temp)
        sim_data = filter_by_closest_temperature(data.get('sim'), target_temp)
        
        # Filter out mixture components from simulation
        if sim_data is not None and 'AllComponents' in sim_data.columns:
            sim_data = sim_data[sim_data['AllComponents'] != 'CO2_N2'].copy()
        
        if exp_co2 is not None:
            exp_co2 = exp_co2.sort_values('Pressure[bar]')
        if exp_n2 is not None:
            exp_n2 = exp_n2.sort_values('Pressure[bar]')
        if sim_data is not None:
            sim_data = sim_data.sort_values('Pressure[bar]')
        
        # First filter experimental data by p_min and p_max
        if exp_co2 is not None:
            exp_co2 = exp_co2[(exp_co2['Pressure[bar]'] >= p_min) & (exp_co2['Pressure[bar]'] <= p_max)]
        if exp_n2 is not None:
            exp_n2 = exp_n2[(exp_n2['Pressure[bar]'] >= p_min) & (exp_n2['Pressure[bar]'] <= p_max)]
        
        # Determine pressure range based on experimental data
        exp_pressures = []
        if exp_co2 is not None and len(exp_co2) > 0:
            exp_pressures.append(exp_co2['Pressure[bar]'].max())
        if exp_n2 is not None and len(exp_n2) > 0:
            exp_pressures.append(exp_n2['Pressure[bar]'].max())
        max_allowed_pressure = min(max(exp_pressures) * 1.1, p_max) if exp_pressures else p_max
        
        # Filter sim and pred data by pressure range
        if sim_data is not None:
            sim_data = sim_data[(sim_data['Pressure[bar]'] >= p_min) & 
                               (sim_data['Pressure[bar]'] <= max_allowed_pressure)]
        co2_pred_filtered = co2_pred[(co2_pred['Pressure[bar]'] >= p_min) & 
                                     (co2_pred['Pressure[bar]'] <= max_allowed_pressure)]
        n2_pred_filtered = n2_pred[(n2_pred['Pressure[bar]'] >= p_min) & 
                                   (n2_pred['Pressure[bar]'] <= max_allowed_pressure)]
        
        # Extract temperatures for labels
        temp_exp_co2 = int(exp_co2['Temperature[K]'].iloc[0]) if exp_co2 is not None and len(exp_co2) > 0 else None
        temp_exp_n2 = int(exp_n2['Temperature[K]'].iloc[0]) if exp_n2 is not None and len(exp_n2) > 0 else None
        temp_sim = int(sim_data['Temperature[K]'].iloc[0]) if sim_data is not None and len(sim_data) > 0 else None
        temp_pred = 298
        
        # Plot CO2 data
        if exp_co2 is not None and len(exp_co2) > 0:
            label = f'CO$_2$ Exp ({temp_exp_co2}K)' if temp_exp_co2 else 'CO$_2$ Exp'
            ax.plot(exp_co2['Pressure[bar]'], exp_co2['Uptake[mol/kg]'],
                   's--', color=colors['CO2']['exp'], linewidth=1.5, markersize=4,
                   alpha=0.8, label=label)
        
        if sim_data is not None:
            sim_co2 = sim_data[sim_data['GasName'] == 'CO2']
            if len(sim_co2) > 0:
                label = f'CO$_2$ Sim ({temp_sim}K)' if temp_sim else 'CO$_2$ Sim'
                ax.plot(sim_co2['Pressure[bar]'], sim_co2['AbsLoading'],
                       'o-', color=colors['CO2']['sim'], linewidth=1.5, markersize=4,
                       label=label)
        
        if len(co2_pred_filtered) > 0:
            label = f'CO$_2$ Pred ({temp_pred}K)'
            ax.plot(co2_pred_filtered['Pressure[bar]'], co2_pred_filtered['PredictedInit'],
                   '^-', color=colors['CO2']['pred'], linewidth=1.5, markersize=5,
                   label=label)
        
        # Plot N2 data
        if exp_n2 is not None and len(exp_n2) > 0:
            label = f'N$_2$ Exp ({temp_exp_n2}K)' if temp_exp_n2 else 'N$_2$ Exp'
            ax.plot(exp_n2['Pressure[bar]'], exp_n2['Uptake[mol/kg]'],
                   's--', color=colors['N2']['exp'], linewidth=1.5, markersize=4,
                   alpha=0.8, label=label)
        
        if sim_data is not None:
            sim_n2 = sim_data[sim_data['GasName'] == 'N2']
            if len(sim_n2) > 0:
                label = f'N$_2$ Sim ({temp_sim}K)' if temp_sim else 'N$_2$ Sim'
                ax.plot(sim_n2['Pressure[bar]'], sim_n2['AbsLoading'],
                       'o-', color=colors['N2']['sim'], linewidth=1.5, markersize=4,
                       label=label)
        
        if len(n2_pred_filtered) > 0:
            label = f'N$_2$ Pred ({temp_pred}K)'
            ax.plot(n2_pred_filtered['Pressure[bar]'], n2_pred_filtered['PredictedInit'],
                   '^-', color=colors['N2']['pred'], linewidth=1.5, markersize=5,
                   label=label)
        
        ax.set_title(f'{mof_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pressure (bar)', fontsize=14)
        ax.set_ylabel('Uptake (mol/kg)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=12)
    
    # Hide unused subplots
    for idx in range(n_mofs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'experimental_mof_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPure component comparison plot saved: {output_path}")


def plot_flue_gas_comparison(df_flue_gas, output_dir):
    """Generate flue gas capture performance comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    methods = ['Direct_ML', 'Exp_IAST', 'GCMC_Mix']
    method_labels = ['MAPP', 'Exp + IAST', 'GCMC']
    cmap_methods = plt.get_cmap("rainbow")
    method_colors = [cmap_methods(i / len(methods)) for i in range(len(methods))]
    
    mof_names = df_flue_gas['MOF'].values
    x = np.arange(len(mof_names))
    width = 0.20
    
    # Plot 1: CO2 Working Capacity
    ax = axes[0, 0]
    for i, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        col = f'{method}_WorkingCapacity'
        if col in df_flue_gas.columns:
            values = df_flue_gas[col].values
            mask = ~pd.isna(values)
            ax.bar(x[mask] + i*width, values[mask], width, 
                  label=label, color=color, alpha=0.8)
    
    ax.set_ylabel(r'$\mathbf{CO_2\ Working\ Capacity\ (mol/kg)}$', fontsize=12, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(mof_names, rotation=0, fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=11)
    
    # Plot 2: Selectivity at Adsorption
    ax = axes[0, 1]
    for i, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        col = f'{method}_Selectivity_Ads'
        if col in df_flue_gas.columns:
            values = df_flue_gas[col].values
            mask = ~pd.isna(values)
            ax.bar(x[mask] + i*width, values[mask], width,
                  label=label, color=color, alpha=0.8)
    
    ax.set_ylabel(r'$\mathbf{CO_2/N_2\ Selectivity}$', fontsize=12, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(mof_names, rotation=0, fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=11)
    
    # Plot 3: CO2 Uptake at Adsorption
    ax = axes[1, 0]
    for i, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        col = f'{method}_CO2_Ads'
        if col in df_flue_gas.columns:
            values = df_flue_gas[col].values
            mask = ~pd.isna(values)
            ax.bar(x[mask] + i*width, values[mask], width,
                  label=label, color=color, alpha=0.8)
    
    ax.set_ylabel(r'$\mathbf{CO_2\ Uptake\ (mol/kg)}$', fontsize=12, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(mof_names, rotation=0, fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=11)
    
    # Plot 4: Performance Scatter
    ax = axes[1, 1]
    annotations_data = []
    for i, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        wc_col = f'{method}_WorkingCapacity'
        sel_col = f'{method}_Selectivity_Ads'
        if wc_col in df_flue_gas.columns and sel_col in df_flue_gas.columns:
            wc_values = df_flue_gas[wc_col].values
            sel_values = df_flue_gas[sel_col].values
            mask = ~(pd.isna(wc_values) | pd.isna(sel_values))
            
            ax.scatter(wc_values[mask], sel_values[mask], 
                      s=150, label=label, color=color, alpha=0.7, 
                      edgecolors='black', linewidth=1.5)
            
            for j, mof in enumerate(mof_names[mask]):
                annotations_data.append({
                    'mof': mof,
                    'x': wc_values[mask][j],
                    'y': sel_values[mask][j],
                    'method': method
                })
    
    # Add annotations with arrows
    if len(annotations_data) > 0:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        offset_x = 0.05 * x_range
        offset_y = 0.05 * y_range
        
        used_positions = []
        for ann_idx, ann in enumerate(annotations_data):
            base_angle = 2 * np.pi * ann_idx / max(1, len(annotations_data))
            chosen_pos = None
            for attempt in range(12):
                angle = base_angle + attempt * (np.pi / 6)
                dx = offset_x * np.cos(angle)
                dy = offset_y * np.sin(angle)
                tx = ann['x'] + dx
                ty = ann['y'] + dy if np.random.random() > 0.5 else ann['y'] - dy
                
                # Check overlap
                overlap = False
                for ux, uy in used_positions:
                    dist = np.sqrt((tx - ux) ** 2 + (ty - uy) ** 2)
                    if dist < 0.025 * max(x_range, y_range):
                        overlap = True
                        break
                if not overlap:
                    chosen_pos = (tx, ty)
                    used_positions.append(chosen_pos)
                    break
            if chosen_pos is None:
                chosen_pos = (ann['x'] + offset_x, ann['y'] + offset_y)
            
            ax.annotate(ann['mof'], xy=(ann['x'], ann['y']), xytext=chosen_pos,
                       fontsize=8, ha='center', va='center',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                     color='dimgray', lw=0.8, alpha=0.4))
    
    ax.set_xlabel(r'$\mathbf{CO_2\ Working\ Capacity\ (mol/kg)}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{CO_2/N_2\ Selectivity}$', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    output_path = output_dir / 'flue_gas_capture_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Flue gas comparison plot saved: {output_path}")


# =============================================================================
# Main Analysis Functions
# =============================================================================

def calculate_flue_gas_metrics(df_co2, df_n2, mof_data, target_temp=298):
    """Calculate flue gas capture performance metrics"""
    if not HAS_PYGAPS:
        print("Warning: pyGAPS not available, skipping IAST calculations")
        return pd.DataFrame()
    
    # Prepare pure component data
    df_co2_pure = df_co2[df_co2['CO2Fraction'] == 1.0].copy()
    df_n2_pure = df_n2[df_n2['CO2Fraction'] == 0.0].copy()
    
    available_mofs = [m for m in mof_data.keys() if m in df_co2_pure['CifId'].values]
    
    # Convenience variables
    ads_p = FLUE_GAS_CONDITIONS['adsorption']['pressure']
    ads_xco2 = FLUE_GAS_CONDITIONS['adsorption']['co2_fraction']
    ads_xn2 = FLUE_GAS_CONDITIONS['adsorption']['n2_fraction']
    des_p = FLUE_GAS_CONDITIONS['desorption']['pressure']
    des_xco2 = FLUE_GAS_CONDITIONS['desorption']['co2_fraction']
    des_xn2 = FLUE_GAS_CONDITIONS['desorption']['n2_fraction']
    
    gas_ratio_ads = ads_xco2 / max(1e-12, ads_xn2)
    gas_ratio_des = des_xco2 / max(1e-12, des_xn2)
    tol = 0.01
    
    results = []
    
    for mof_name in available_mofs:
        print(f"\n{'='*80}")
        print(f"Processing {mof_name} for flue gas conditions")
        print(f"{'='*80}")
        
        data = mof_data[mof_name]
        result = {'MOF': mof_name}
        
        # Filter by temperature
        exp_co2 = filter_by_closest_temperature(data.get('exp_CO2'), target_temp)
        exp_n2 = filter_by_closest_temperature(data.get('exp_N2'), target_temp)
        sim_data = filter_by_closest_temperature(data.get('sim'), target_temp)
        
        # Separate pure and mixture simulation data
        sim_pure = None
        sim_mix = None
        if sim_data is not None and len(sim_data) > 0 and 'AllComponents' in sim_data.columns:
            sim_pure = sim_data[sim_data['AllComponents'] != 'CO2_N2'].copy()
            sim_mix = sim_data[sim_data['AllComponents'] == 'CO2_N2'].copy()
        else:
            sim_pure = sim_data
        
        # Get pure component predictions
        co2_pred = df_co2_pure[df_co2_pure['CifId'] == mof_name].copy()
        n2_pred = df_n2_pure[df_n2_pure['CifId'] == mof_name].copy()
        
        # Method 1: Direct ML Prediction (mixture composition)
        print("\nMethod 1: Direct ML Prediction (Mixture)")
        mof_co2_mix = df_co2[df_co2['CifId'] == mof_name].copy()
        mof_n2_mix = df_n2[df_n2['CifId'] == mof_name].copy()
        
        if len(mof_co2_mix) > 0 and len(mof_n2_mix) > 0:
            co2_ads_direct = mof_co2_mix[
                (abs(mof_co2_mix['Pressure[bar]'] - ads_p) < tol) &
                (abs(mof_co2_mix['CO2Fraction'] - ads_xco2) < tol)
            ]
            n2_ads_direct = mof_n2_mix[
                (abs(mof_n2_mix['Pressure[bar]'] - ads_p) < tol) &
                (abs(mof_n2_mix['CO2Fraction'] - ads_xco2) < tol)
            ]
            co2_des_direct = mof_co2_mix[
                (abs(mof_co2_mix['Pressure[bar]'] - des_p) < tol) &
                (abs(mof_co2_mix['CO2Fraction'] - des_xco2) < tol)
            ]
            n2_des_direct = mof_n2_mix[
                (abs(mof_n2_mix['Pressure[bar]'] - des_p) < tol) &
                (abs(mof_n2_mix['CO2Fraction'] - des_xco2) < tol)
            ]
            
            if len(co2_ads_direct) > 0 and len(n2_ads_direct) > 0 and \
               len(co2_des_direct) > 0 and len(n2_des_direct) > 0:
                co2_ads_ml = symlog_inverse(co2_ads_direct['Predicted'].iloc[0])
                n2_ads_ml = symlog_inverse(n2_ads_direct['Predicted'].iloc[0])
                co2_des_ml = symlog_inverse(co2_des_direct['Predicted'].iloc[0])
                n2_des_ml = symlog_inverse(n2_des_direct['Predicted'].iloc[0])
                
                result['Direct_ML_CO2_Ads'] = co2_ads_ml
                result['Direct_ML_N2_Ads'] = n2_ads_ml
                result['Direct_ML_CO2_Des'] = co2_des_ml
                result['Direct_ML_WorkingCapacity'] = co2_ads_ml - co2_des_ml
                result['Direct_ML_Selectivity_Ads'] = (co2_ads_ml / max(1e-12, n2_ads_ml)) / gas_ratio_ads
                result['Direct_ML_Selectivity_Des'] = (co2_des_ml / max(1e-12, n2_des_ml)) / gas_ratio_des
                print(f"  ✓ Working Capacity: {result['Direct_ML_WorkingCapacity']:.4f} mol/kg")
                print(f"  ✓ Selectivity (Ads): {result['Direct_ML_Selectivity_Ads']:.2f}")
        
        # Method 2: Experimental + IAST
        print("\nMethod 2: Experimental Data + IAST")
        if exp_co2 is not None and exp_n2 is not None and len(exp_co2) >= 3 and len(exp_n2) >= 3:
            iso_co2_exp = fit_pure_isotherms_from_data(
                exp_co2['Pressure[bar]'].values, exp_co2['Uptake[mol/kg]'].values,
                'CO2', mof_name, target_temp
            )
            iso_n2_exp = fit_pure_isotherms_from_data(
                exp_n2['Pressure[bar]'].values, exp_n2['Uptake[mol/kg]'].values,
                'N2', mof_name, target_temp
            )
            
            if iso_co2_exp and iso_n2_exp:
                co2_ads_iast, n2_ads_iast = calculate_iast_loading(iso_co2_exp, iso_n2_exp, ads_p, ads_xco2)
                co2_des_iast, n2_des_iast = calculate_iast_loading(iso_co2_exp, iso_n2_exp, des_p, des_xco2)
                
                if co2_ads_iast and co2_des_iast:
                    result['Exp_IAST_CO2_Ads'] = co2_ads_iast
                    result['Exp_IAST_WorkingCapacity'] = co2_ads_iast - co2_des_iast
                    result['Exp_IAST_Selectivity_Ads'] = (co2_ads_iast / max(1e-12, n2_ads_iast)) / gas_ratio_ads
                    result['Exp_IAST_Selectivity_Des'] = (co2_des_iast / max(1e-12, n2_des_iast)) / gas_ratio_des
                    print(f"  ✓ Working Capacity: {result['Exp_IAST_WorkingCapacity']:.4f} mol/kg")
                    print(f"  ✓ Selectivity (Ads): {result['Exp_IAST_Selectivity_Ads']:.2f}")
        
        # Method 3: GCMC Pure + IAST
        print("\nMethod 3: GCMC Simulation + IAST")
        if sim_pure is not None and len(sim_pure) > 0:
            sim_co2 = sim_pure[sim_pure['GasName'] == 'CO2']
            sim_n2 = sim_pure[sim_pure['GasName'] == 'N2']
            
            if len(sim_co2) >= 3 and len(sim_n2) >= 3:
                iso_co2_sim = fit_pure_isotherms_from_data(
                    sim_co2['Pressure[bar]'].values, sim_co2['AbsLoading'].values,
                    'CO2', mof_name, target_temp
                )
                iso_n2_sim = fit_pure_isotherms_from_data(
                    sim_n2['Pressure[bar]'].values, sim_n2['AbsLoading'].values,
                    'N2', mof_name, target_temp
                )
                
                if iso_co2_sim and iso_n2_sim:
                    co2_ads_iast, n2_ads_iast = calculate_iast_loading(iso_co2_sim, iso_n2_sim, ads_p, ads_xco2)
                    co2_des_iast, n2_des_iast = calculate_iast_loading(iso_co2_sim, iso_n2_sim, des_p, des_xco2)
                    
                    if co2_ads_iast and co2_des_iast:
                        result['Sim_IAST_CO2_Ads'] = co2_ads_iast
                        result['Sim_IAST_WorkingCapacity'] = co2_ads_iast - co2_des_iast
                        result['Sim_IAST_Selectivity_Ads'] = (co2_ads_iast / max(1e-12, n2_ads_iast)) / gas_ratio_ads
                        print(f"  ✓ Working Capacity: {result['Sim_IAST_WorkingCapacity']:.4f} mol/kg")
        
        # Method 4: GCMC Direct Mixture
        print("\nMethod 4: GCMC Direct Mixture")
        if sim_mix is not None and len(sim_mix) > 0:
            mix_ads = sim_mix[
                (abs(sim_mix['Pressure[bar]'] - ads_p) < tol) &
                (abs(sim_mix['CO2Fraction'] - ads_xco2) < tol)
            ]
            mix_des = sim_mix[
                (abs(sim_mix['Pressure[bar]'] - des_p) < tol) &
                (abs(sim_mix['CO2Fraction'] - des_xco2) < tol)
            ]
            
            if len(mix_ads) >= 2 and len(mix_des) >= 2:
                co2_ads_gcmc = mix_ads[mix_ads['GasName'] == 'CO2']['AbsLoading'].iloc[0]
                n2_ads_gcmc = mix_ads[mix_ads['GasName'] == 'N2']['AbsLoading'].iloc[0]
                co2_des_gcmc = mix_des[mix_des['GasName'] == 'CO2']['AbsLoading'].iloc[0]
                n2_des_gcmc = mix_des[mix_des['GasName'] == 'N2']['AbsLoading'].iloc[0]
                
                result['GCMC_Mix_CO2_Ads'] = co2_ads_gcmc
                result['GCMC_Mix_WorkingCapacity'] = co2_ads_gcmc - co2_des_gcmc
                result['GCMC_Mix_Selectivity_Ads'] = (co2_ads_gcmc / max(1e-12, n2_ads_gcmc)) / gas_ratio_ads
                result['GCMC_Mix_Selectivity_Des'] = (co2_des_gcmc / max(1e-12, n2_des_gcmc)) / gas_ratio_des
                print(f"  ✓ Working Capacity: {result['GCMC_Mix_WorkingCapacity']:.4f} mol/kg")
                print(f"  ✓ Selectivity (Ads): {result['GCMC_Mix_Selectivity_Ads']:.2f}")
        
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main function for experimental MOF comparison"""
    warnings.filterwarnings('ignore')
    
    # Setup output directory
    results_dir = PROJECT_ROOT / config["output"]["results"]
    figs_dir = PROJECT_ROOT / config["output"]["figures"]
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("EXPERIMENTAL MOF VALIDATION")
    print("="*80)
    
    # Load data
    print("\nLoading prediction data...")
    df_co2, df_n2 = load_prediction_data(args.model_name)
    
    print("\nLoading experimental and simulation data...")
    mof_data = load_experimental_data()
    
    # Generate pure component comparison plot
    print("\nGenerating pure component comparison plot...")
    plot_pure_component_comparison(df_co2, df_n2, mof_data, figs_dir)
    
    # Calculate flue gas metrics
    print("\nCalculating flue gas capture metrics...")
    df_flue_gas = calculate_flue_gas_metrics(df_co2, df_n2, mof_data)
    
    if len(df_flue_gas) > 0:
        # Save results
        output_csv = results_dir / f"experimental_mof_flue_gas_metrics_{args.model_name}.csv"
        df_flue_gas.to_csv(output_csv, index=False)
        print(f"\nFlue gas metrics saved: {output_csv}")
        
        # Generate flue gas comparison plot
        print("\nGenerating flue gas comparison plot...")
        plot_flue_gas_comparison(df_flue_gas, figs_dir)
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    # Extract default model name from config
    default_mixture_model = config["models"]["MAPP_GMOF"]
    default_model_name = default_mixture_model.split('/')[-2] + "_" + default_mixture_model.split('/')[-1]
    
    parser = argparse.ArgumentParser(
        description='Compare ML predictions with experimental data for selected MOFs'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=default_model_name,
        help=f'Model name for prediction files (default from config: {default_model_name})'
    )
    
    args = parser.parse_args()
    main(args)
