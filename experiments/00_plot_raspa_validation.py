#!/usr/bin/env python3
"""
RASPA2 Validation Combined Plot Script

This script generates combined validation plots comparing RASPA2 simulation results
with experimental data for multiple MOFs. Includes CO2 and N2 isotherms for:
- Standard MOFs (CALF20, IRMOF1, UiO66)
- MIL-53(Al) with open/closed pore state comparison

Usage:
    python 00_plot_raspa_validation.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

# Project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# MOFs to include in individual subplots
MOFS = ['CALF20', 'IRMOF1', 'UiO66']

# Pressure range
P_MIN = 0
P_MAX = 10  # bar


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data(mof_dir: Path):
    """
    Load simulation and experimental data for a MOF.
    
    Args:
        mof_dir: Path to MOF data directory
        
    Returns:
        Dictionary with keys 'sim', 'exp_CO2', 'exp_N2' containing DataFrames
    """
    sim_file = mof_dir / 'simulation_data.csv'
    exp_co2_file = mof_dir / 'experimental_data_CO2.csv'
    exp_n2_file = mof_dir / 'experimental_data_N2.csv'
    
    data = {'sim': None, 'exp_CO2': None, 'exp_N2': None}
    
    if sim_file.exists():
        data['sim'] = pd.read_csv(sim_file)
    if exp_co2_file.exists():
        data['exp_CO2'] = pd.read_csv(exp_co2_file)
    if exp_n2_file.exists():
        data['exp_N2'] = pd.read_csv(exp_n2_file)
    
    return data


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_mof_subplot(ax, mof_name, data, colors):
    """
    Plot a single MOF's data on the given axes.
    
    Args:
        ax: Matplotlib axis object
        mof_name: Name of the MOF
        data: Dictionary containing simulation and experimental data
        colors: Dictionary specifying colors for different gases and data types
    """
    # Get unique temperatures from simulation data
    temps = []
    if data['sim'] is not None:
        temps = sorted(data['sim']['Temperature[K]'].unique())
    
    # If no simulation temps, try to get from experimental data
    if not temps:
        for gas in ['CO2', 'N2']:
            df_exp = data.get(f'exp_{gas}')
            if df_exp is not None and 'Temperature[K]' in df_exp.columns:
                temps = sorted(df_exp['Temperature[K]'].unique())
                break
    
    # If still no temps, use default
    if not temps:
        temps = [298]
    
    # Use only the first temperature for simplicity (most MOFs have single temp)
    # For multi-temp MOFs like CALF20, we'll use the first temp
    temp = temps[0]
    
    # Plot simulation data for this temperature
    if data['sim'] is not None:
        df_sim = data['sim']
        for gas in ['CO2', 'N2']:
            df_gas = df_sim[(df_sim['GasName'] == gas) & 
                            (abs(df_sim['Temperature[K]'] - temp) < 5)].copy()
            df_gas = df_gas[(df_gas['Pressure[bar]'] >= P_MIN) & 
                            (df_gas['Pressure[bar]'] <= P_MAX) &
                            (df_gas['MoleFraction']==1)
                            ]
            df_gas = df_gas.sort_values('Pressure[bar]')
            
            if len(df_gas) > 0:
                actual_temp = df_gas['Temperature[K]'].iloc[0]
                ax.plot(
                    df_gas['Pressure[bar]'],
                    df_gas['AbsLoading'],
                    'o-',
                    color=colors[gas]['sim'],
                    linewidth=2,
                    markersize=5,
                    label=f'{gas.replace("2", "$_2$")} Sim {actual_temp:.0f}K'
                )
    
    # Plot experimental data for this temperature
    for gas in ['CO2', 'N2']:
        df_exp = data.get(f'exp_{gas}')
        if df_exp is not None:
            # Filter by temperature if available
            if 'Temperature[K]' in df_exp.columns:
                df_filtered = df_exp[abs(df_exp['Temperature[K]'] - temp) < 5].copy()
            else:
                df_filtered = df_exp.copy()
            
            df_filtered = df_filtered[(df_filtered['Pressure[bar]'] >= P_MIN) & 
                                      (df_filtered['Pressure[bar]'] <= P_MAX)]
            df_filtered = df_filtered.sort_values('Pressure[bar]')
            
            if len(df_filtered) > 0:
                if 'Temperature[K]' in df_filtered.columns:
                    actual_temp = df_filtered['Temperature[K]'].mean()
                else:
                    actual_temp = 298
                
                ax.plot(
                    df_filtered['Pressure[bar]'],
                    df_filtered['Uptake[mol/kg]'],
                    's--',
                    color=colors[gas]['exp'],
                    linewidth=1.5,
                    markersize=4,
                    alpha=0.8,
                    label=f'{gas.replace("2", "$_2$")} Exp {actual_temp:.0f}K'
                )
    
    # Formatting
    ax.set_xlabel('Pressure (bar)', fontsize=11)
    ax.set_ylabel('Loading (mol/kg)', fontsize=11)
    ax.set_title(mof_name, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(which='both', direction='in')


def plot_mil53_comparison(ax, base_dir):
    """
    Plot MIL-53(Al) open vs closed pore comparison.
    
    Args:
        ax: Matplotlib axis object
        base_dir: Base directory containing MOF data
    """
    # Load data for both pore states
    data_a = load_data(base_dir / 'MIL53Ala')  # Closed pore
    data_b = load_data(base_dir / 'MIL53Alb')  # Open pore
    
    # Color scheme using rainbow colormap for different states
    cmap = plt.get_cmap("rainbow")
    n_colors = 6
    state_colors = [cmap(i / n_colors) for i in range(n_colors)]
    colors_closed = {'CO2': state_colors[0], 'N2': state_colors[1]}  # Closed pore
    colors_open = {'CO2': state_colors[2], 'N2': state_colors[3]}    # Open pore
    
    # Plot simulation data for closed pore (MIL53Ala)
    if data_a['sim'] is not None:
        df_sim = data_a['sim']
        for gas in ['CO2', 'N2']:
            df_gas = df_sim[df_sim['GasName'] == gas].copy()
            df_gas = df_gas[(df_gas['Pressure[bar]'] >= P_MIN) & 
                            (df_gas['Pressure[bar]'] <= P_MAX)]
            df_gas = df_gas.sort_values('Pressure[bar]')
            if 'Temperature[K]' in df_gas.columns:
                temp = df_gas['Temperature[K]'].mean()
            
            if len(df_gas) > 0:
                ax.plot(
                    df_gas['Pressure[bar]'],
                    df_gas['AbsLoading'],
                    'o-',
                    color=colors_closed[gas],
                    linewidth=2,
                    markersize=5,
                    label=f'{gas.replace("2", "$_2$")} sim Closed {temp:.0f}K'
                )
    
    # Plot simulation data for open pore (MIL53Alb)
    if data_b['sim'] is not None:
        df_sim = data_b['sim']
        for gas in ['CO2', 'N2']:
            df_gas = df_sim[df_sim['GasName'] == gas].copy()
            df_gas = df_gas[(df_gas['Pressure[bar]'] >= P_MIN) & 
                            (df_gas['Pressure[bar]'] <= P_MAX)]
            df_gas = df_gas.sort_values('Pressure[bar]')
            if 'Temperature[K]' in df_gas.columns:
                temp = df_gas['Temperature[K]'].mean()
            
            if len(df_gas) > 0:
                ax.plot(
                    df_gas['Pressure[bar]'],
                    df_gas['AbsLoading'],
                    '^-',
                    color=colors_open[gas],
                    linewidth=2,
                    markersize=5,
                    label=f'{gas.replace("2", "$_2$")} sim Open {temp:.0f}K'
                )
    
    # Plot experimental data (same for both - use MIL53Ala's exp data)
    colors_exp = {'CO2': state_colors[4], 'N2': state_colors[5]}
    for gas in ['CO2', 'N2']:
        df_exp = data_a.get(f'exp_{gas}')
        if df_exp is not None:
            df_filtered = df_exp[(df_exp['Pressure[bar]'] >= P_MIN) & 
                                 (df_exp['Pressure[bar]'] <= P_MAX)].copy()
            df_filtered = df_filtered.sort_values('Pressure[bar]')
            
            if len(df_filtered) > 0:
                if 'Temperature[K]' in df_filtered.columns:
                    temp = df_filtered['Temperature[K]'].mean()
                else:
                    temp = 298
                
                ax.plot(
                    df_filtered['Pressure[bar]'],
                    df_filtered['Uptake[mol/kg]'],
                    's--',
                    color=colors_exp[gas],
                    linewidth=1.5,
                    markersize=4,
                    alpha=0.8,
                    label=f'{gas.replace("2", "$_2$")} Exp {temp:.0f}K'
                )
    
    # Formatting
    ax.set_xlabel('Pressure (bar)', fontsize=11)
    ax.set_ylabel('Loading (mol/kg)', fontsize=11)
    ax.set_title('MIL-53(Al): Open vs Closed Pore', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(which='both', direction='in')


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to generate RASPA validation plots."""
    # Base directory for RASPA validation data
    base_dir = PROJECT_ROOT / config["data"]["exp_mof_data"]
    
    # Output directory for figures
    fig_dir = PROJECT_ROOT / config["output"]["figures"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Set theme
    sns.set_theme(style="white", font_scale=1.0)
    
    # Create figure with 2x2 subplots (3 individual MOFs + 1 MIL53 comparison)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Color scheme using rainbow colormap
    cmap = plt.get_cmap("rainbow")
    n_colors = 4
    base_colors = [cmap(i / n_colors) for i in range(n_colors)]
    colors = {
        'CO2': {'sim': base_colors[0], 'exp': base_colors[1]},
        'N2': {'sim': base_colors[2], 'exp': base_colors[3]}
    }
    
    # Plot individual MOFs
    print("=" * 60)
    print("Generating RASPA2 Validation Plots")
    print("=" * 60)
    
    for idx, mof_name in enumerate(MOFS):
        ax = axes[idx]
        mof_dir = base_dir / mof_name
        
        if not mof_dir.exists():
            print(f"Warning: {mof_name} directory not found at {mof_dir}")
            continue
        
        print(f"Processing {mof_name}...")
        data = load_data(mof_dir)
        plot_mof_subplot(ax, mof_name, data, colors)
    
    # Plot MIL-53(Al) comparison in the last subplot
    print("Processing MIL-53(Al) open/closed comparison...")
    plot_mil53_comparison(axes[3], base_dir)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = fig_dir / 'raspa_validation_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    
    print(f"\nSaved validation plot to: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
