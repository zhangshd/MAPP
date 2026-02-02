#!/usr/bin/env python
"""
MAPP vs ML-Pure+IAST Speed Benchmark (PSA Grid Simulation)

Scenario:
Simulate a PSA process where thousands of (Pressure, CO2Fraction) conditions 
need to be queried for each MOF.

Comparison:
1. MAPP: Direct batch inference for all conditions on GPU.
2. ML-Pure+IAST: Pure component prediction (once) + Fitting (once) + IAST calculation (loop over all conditions).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import functools
import random as rand_module

warnings.filterwarnings("ignore")

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "MOFTransformer"))

import pygaps as pg
import pygaps.modelling as pgm
import pygaps.iast as pgi


# ============================================================================
# Configuration
# ============================================================================

# Default model paths
DEFAULT_MIXTURE_MODEL = "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/version_14"
DEFAULT_PURE_MODEL = "MOFTransformer/logs/ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer/version_2"
DEFAULT_CIF_DIR = "GCMC/data/ddmof/cifs"

# Pressure points for pure component isotherms (for fitting)
PURE_PRESSURE_POINTS = [
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
]


# ============================================================================
# Utility Functions
# ============================================================================

def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog transform: sign(y) * threshold * (10^|y| - 1)"""
    return np.sign(y) * threshold * (10 ** np.abs(y) - 1)


def generate_psa_grid(n_pressures, n_fractions, p_min=0.01, p_max=10.0):
    """
    Generate a grid of (Pressure, CO2Fraction) conditions for PSA simulation.
    
    Args:
        n_pressures (int): Number of pressure points (log-spaced)
        n_fractions (int): Number of composition points (linearly spaced)
        p_min, p_max: Pressure range in bar
        
    Returns:
        pressures (np.array): Flattened array of pressures
        fractions (np.array): Flattened array of CO2 fractions
    """
    p_range = np.logspace(np.log10(p_min), np.log10(p_max), n_pressures)
    y_range = np.linspace(0.01, 0.99, n_fractions)
    
    # Create meshgrid
    P, Y = np.meshgrid(p_range, y_range)
    
    return P.flatten(), Y.flatten()


# ============================================================================
# MAPP Benchmark Module
# ============================================================================

def benchmark_mapp_grid(cif_list, mixture_model_dir, output_dir, ps_grid, ys_grid, n_runs=1):
    """
    Benchmark MAPP direct mixture prediction on a dense grid.
    
    Args:
        ps_grid: Array of pressure points
        ys_grid: Array of CO2 fraction points
    """
    from inference import load_model_from_dir, InferenceDataset
    from torch.utils.data import DataLoader
    import torch
    
    mixture_model_dir = PROJECT_ROOT / mixture_model_dir
    saved_dir = output_dir / "mapp_inference"
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    timings = {"model_load": [], "data_prep": [], "inference": [], "total": []}
    
    n_conditions = len(ps_grid)
    print(f"  Grid size: {n_conditions} points/MOF x {len(cif_list)} MOFs = {n_conditions * len(cif_list)} total inferences")
    
    for run_idx in range(n_runs):
        print(f"  MAPP Run {run_idx + 1}/{n_runs}...")
        
        t_start = time.time()
        
        # 1. Model Loading
        t_load_start = time.time()
        model, trainer = load_model_from_dir(mixture_model_dir)
        model.eval()
        t_load = time.time() - t_load_start
        timings["model_load"].append(t_load)
        
        # 2. Data Preparation
        t_prep_start = time.time()
        # Create dataset with full grid
        # InferenceDataset expects list logic: for each CIF, combine with press and co2frac
        # We need to pass lists. Since press and co2frac arguments in InferenceDataset 
        # create valid combinations, we can pass our grid directly if we structure it right.
        # But InferenceDataset usually does outer product. To do specific pairs, use 'inputs'.
        
        # Construct inputs manually to avoid huge outer product if we just passed P and Y lists
        manual_inputs = []
        condi_cols = model.hparams["config"].get("condi_cols", [])
        
        # Determine Pressure format
        use_arcsinh = len(condi_cols) > 0 and "Arcsinh" in condi_cols[0]
        use_symlog = len(condi_cols) > 1 and "Symlog" in condi_cols[1]
        
        # Prepare inputs list: [cif_name, p_val, (p_val_2), y_val]
        # This is computationally expensive in Python loop for millions of points.
        # But for benchmark fair comparison, data prep is part of the flow.
        # Optimized: Batch preparation
        

        
        # HACK: Manually inject inputs to avoid Cartesian product of flattened arrays
        # If we pass P and Y arrays to InferenceDataset, it loops P then Y.
        # Here we have matched pairs. We must construct 'inputs' list manually.
        # Format: [cif_id, press_trans_1, (press_trans_2), co2_frac]
        
        inputs = []
        symlog_threshold = 1e-4
        
        # Vectorized prep could be faster but let's stick to simple loop for now
        # OR better: Assume we want Cartesian Product of unique P and unique Y?
        # Yes, generate_psa_grid does Cartesian product. 
        # So we can just extract unique P and Y axes and pass to InferenceDataset!
        unique_p = np.unique(ps_grid)
        unique_y = np.unique(ys_grid)
        
        # Update dataset with axes
        infer_dataset = InferenceDataset(
            cif_list, 
            co2frac=unique_y,
            press=unique_p, 
            saved_dir=saved_dir,
            clean=True, 
            **model.hparams["config"]
        )
        
        infer_dataset.setup()
        
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=min(len(infer_dataset), model.hparams["config"].get("per_gpu_batchsize", 64)), # Increase batch size
            shuffle=False,
            num_workers=model.hparams["config"].get("num_workers", 4),
            collate_fn=functools.partial(InferenceDataset.collate, 
                                         img_size=model.hparams["config"].get("img_size", 30),
                                         task_num=len(model.hparams["config"]["tasks"]))
        )
        t_prep = time.time() - t_prep_start
        timings["data_prep"].append(t_prep)
        
        # 3. Inference
        t_infer_start = time.time()
        _ = trainer.predict(model, infer_loader)
        t_infer = time.time() - t_infer_start
        timings["inference"].append(t_infer)
        
        t_total = time.time() - t_start
        timings["total"].append(t_total)
    
    return {
        "timings": {k: np.mean(v) for k, v in timings.items()},
        "timings_std": {k: np.std(v) for k, v in timings.items()},
        "n_inferences": len(cif_list) * n_conditions
    }


# ============================================================================
# ML-Pure + IAST Benchmark Module
# ============================================================================

def fit_pure_isotherms_v2(mof_name, df_mof, gas_col_map):
    """(Same as before) Fit pure component isotherms."""
    isotherms_models = []
    
    for gas in ['CO2', 'N2']:
        loading_col = gas_col_map.get(gas)
        if loading_col is None:
            return None
        if loading_col not in df_mof.columns:
            return None
        
        pure_data = df_mof[['Pressure', loading_col]].copy().dropna()
        pure_data.columns = ['Pressure', 'AbsLoading']
        pure_data = pure_data[pure_data['AbsLoading'] > 0].sort_values('Pressure')
        
        if len(pure_data) < 3:
            return None
        
        try:
            isotherm = pg.PointIsotherm(
                isotherm_data=pure_data,
                pressure_key='Pressure', loading_key='AbsLoading',
                material=mof_name, adsorbate=gas, temperature=25,
                pressure_mode='absolute', pressure_unit='bar',
                loading_basis='molar', loading_unit='mmol',
                material_basis='mass', material_unit='g', temperature_unit='°C'
            )
            # Try simpler models first for speed/stability
            model = pgm.model_iso(isotherm, model=["Langmuir", "DSLangmuir", "Henry"], verbose=False)
            isotherms_models.append(model)
        except Exception:
            return None
    
    return isotherms_models


def benchmark_ml_iast_grid(cif_list, pure_model_dir, output_dir, ps_grid, ys_grid, n_runs=1):
    """
    Benchmark ML-Pure + IAST on a dense grid.
    
    Steps:
    1. Load Pure Model
    2. Infer Pure Components (small number of points)
    3. Fit Isotherms
    4. IAST Loop: Solve for every (P, y) point in the grid for every MOF.
    """
    from inference import load_model_from_dir, InferenceDataset
    from torch.utils.data import DataLoader
    import torch
    
    pure_model_dir = PROJECT_ROOT / pure_model_dir
    saved_dir = output_dir / "pure_inference"
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    timings = {"model_load": [], "pure_infer": [], "fitting": [], "iast_loop": [], "total": []}
    
    n_grid_points = len(ps_grid)
    
    for run_idx in range(n_runs):
        print(f"  ML+IAST Run {run_idx + 1}/{n_runs}...")
        t_start = time.time()
        
        # 1. Load Model
        t_load_start = time.time()
        model, trainer = load_model_from_dir(pure_model_dir)
        model.eval()
        t_load = time.time() - t_load_start
        timings["model_load"].append(t_load)
        
        # 2. Pure Inference (Fixed small set of pressure points)
        t_infer_start = time.time()
        # Check pure model type
        condi_cols = model.hparams["config"].get("condi_cols", [])
        has_co2frac = any("CO2Fraction" in col for col in condi_cols)
        
        if has_co2frac:
            dataset_args = {"co2frac": [0.0, 1.0], "press": PURE_PRESSURE_POINTS}
        else:
            dataset_args = {"co2frac": None, "press": PURE_PRESSURE_POINTS}
            
        infer_dataset = InferenceDataset(
            cif_list, saved_dir=saved_dir, clean=True,
            **dataset_args, **model.hparams["config"]
        )
        infer_dataset.setup()
        
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=min(len(infer_dataset), 64),
            shuffle=False, num_workers=2,
            collate_fn=functools.partial(InferenceDataset.collate, 
                                         img_size=model.hparams["config"].get("img_size", 30),
                                         task_num=len(model.hparams["config"]["tasks"]))
        )
        
        outputs = trainer.predict(model, infer_loader)
        t_infer = time.time() - t_infer_start
        timings["pure_infer"].append(t_infer)
        
        # Parse Predictions
        pred_data = {"CifId": [], "Pressure": []}
        for d in infer_dataset:
            pred_data["CifId"].append(d["cif_id"])
            if len(condi_cols) > 0 and "Arcsinh" in condi_cols[0]:
                pred_data["Pressure"].append(np.sinh(d["extra_fea"][0].item()))
            else:
                pred_data["Pressure"].append(10**(d["extra_fea"][0].item()) - 1e-5)
                
        gas_col_map = {}
        for task in model.hparams["config"].get("tasks", {}):
            preds = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
            pred_data[task] = symlog_inverse(preds)
            if "CO2" in task: gas_col_map["CO2"] = task
            elif "N2" in task: gas_col_map["N2"] = task
            
        df_pure = pd.DataFrame(pred_data)
        
        # 3. Fitting
        t_fit_start = time.time()
        fitted_models = {}
        mof_names = df_pure["CifId"].unique()
        
        for mof in mof_names:
            mof_data = df_pure[df_pure["CifId"] == mof]
            models = fit_pure_isotherms_v2(mof, mof_data, gas_col_map)
            if models is not None:
                fitted_models[mof] = models
        t_fit = time.time() - t_fit_start
        timings["fitting"].append(t_fit)
        
        # 4. IAST Loop (The Heavy Part)
        t_iast_start = time.time()
        
        # We need to solve for (P, y) for EACH fitted MOF
        # Total calculations = n_fitted_mofs * n_grid_points
        n_calcs = len(fitted_models) * n_grid_points
        if run_idx == 0:
            print(f"    Starting IAST loop: {len(fitted_models)} MOFs x {n_grid_points} points = {n_calcs} calculations")
        
        # Optimization: Prepare fraction arrays for all points
        # pgi.iast_point_fraction takes (isotherms, mole_fractions, pressure)
        # But it's designed for scalar inputs usually.
        # We will loop explicitly to simulate real usage patterns
        # (though pygaps might have some vectorization, IAST iterative solver is hard to vectorize fully)
        
        for mof, models in fitted_models.items():
            # For each MOF, calculate all grid points
            # We can use pgi.iast_point_fraction in a loop
            # Or we can try to optimize if possible, but let's stick to loop for realism of standard tools
            
            for p, y in zip(ps_grid, ys_grid):
                try:
                    # y is CO2 fraction. fractions = [y_CO2, y_N2]
                    _ = pgi.iast_point_fraction(
                        models, [y, 1-y], p, verbose=False
                    )
                except Exception:
                    pass
        
        t_iast = time.time() - t_iast_start
        timings["iast_loop"].append(t_iast)
        
        t_total = time.time() - t_start
        timings["total"].append(t_total)
    
    return {
        "timings": {k: np.mean(v) for k, v in timings.items()},
        "timings_std": {k: np.std(v) for k, v in timings.items()},
        "n_calcs": len(fitted_models) * n_grid_points
    }


# ============================================================================
# Visualization & Main
# ============================================================================

def plot_benchmark_results(mapp_res, iast_res, output_dir, n_mofs, n_points):
    """Plot benchmark results focusing on Query Throughput."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Total Time
    ax1 = axes[0]
    methods = ['MAPP', 'ML-Pure+IAST']
    times = [mapp_res["timings"]["total"], iast_res["timings"]["total"]]
    
    bars = ax1.bar(methods, times, color=['#2ecc71', '#e74c3c'], width=0.6)
    ax1.set_ylabel('Total Time (s)', fontsize=12)
    ax1.set_title(f'Time to Process {n_mofs} MOFs × {n_points} Points\nTotal Queries: {n_mofs*n_points}', fontsize=12)
    
    # Labels
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.1f}s', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Speedup
    speedup = iast_res["timings"]["total"] / max(mapp_res["timings"]["total"], 0.01)
    ax1.text(0.5, 0.9, f'Speedup: {speedup:.1f}×', transform=ax1.transAxes, 
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 2. Stacked Breakdown
    ax2 = axes[1]
    
    # MAPP Components
    mapp_comps = [mapp_res["timings"]["model_load"], mapp_res["timings"]["data_prep"], mapp_res["timings"]["inference"]]
    mapp_labels = ["Model Load", "Data Prep", "Inference"]
    
    # IAST Components
    iast_comps = [iast_res["timings"]["model_load"], iast_res["timings"]["pure_infer"], 
                  iast_res["timings"]["fitting"], iast_res["timings"]["iast_loop"]]
    iast_labels = ["Model Load", "Pure Infer", "Fitting", "IAST Loop"]
    
    # Plot MAPP stack
    bottom = 0
    for t, l in zip(mapp_comps, mapp_labels):
        ax2.bar(0, t, bottom=bottom, label=f"MAPP: {l}", width=0.5)
        bottom += t
        
    # Plot IAST stack
    bottom = 0
    for t, l in zip(iast_comps, iast_labels):
        ax2.bar(1, t, bottom=bottom, label=f"IAST: {l}", width=0.5)
        # Add text for IAST loop if significant
        if l == "IAST Loop" and t > 5:
             ax2.text(1, bottom + t/2, "IAST Loop", ha='center', va='center', color='white', fontweight='bold')
        bottom += t
        
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['MAPP', 'ML-Pure+IAST'])
    ax2.set_title('Time Breakdown', fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "grid_benchmark_comparison.png", dpi=150)
    print(f"Plot saved to {output_dir / 'grid_benchmark_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='PSA Grid Benchmark: MAPP vs IAST')
    parser.add_argument('--n_mofs', type=int, default=5, help='Number of MOFs')
    parser.add_argument('--n_pressures', type=int, default=50, help='Number of pressure points')
    parser.add_argument('--n_fractions', type=int, default=20, help='Number of CO2 fraction points')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of repeats')
    parser.add_argument('--random', action='store_true', help='Random sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cif_dir', type=str, default=DEFAULT_CIF_DIR)
    parser.add_argument('--output_dir', type=str, default='results/benchmark_psa')
    
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. MOF Selection
    cif_dir = PROJECT_ROOT / args.cif_dir
    all_cifs = list(cif_dir.glob("*.cif"))
    if args.random:
        rand_module.seed(args.seed)
        cif_list = rand_module.sample(all_cifs, min(len(all_cifs), args.n_mofs))
    else:
        cif_list = all_cifs[:args.n_mofs]
        
    # 2. Grid Generation
    P, Y = generate_psa_grid(args.n_pressures, args.n_fractions)
    n_points = len(P)
    
    print(f"\n{'='*60}")
    print(f"PSA GRID QUERY BENCHMARK")
    print(f"{'='*60}")
    print(f"MOFs: {len(cif_list)}")
    print(f"Grid: {args.n_pressures} Pressures x {args.n_fractions} Fractions = {n_points} Points/MOF")
    print(f"Total Queries: {len(cif_list) * n_points}")
    print(f"{'='*60}\n")
    
    # 3. Run Benchmarks
    print("Running MAPP Grid Inference...")
    mapp_res = benchmark_mapp_grid(cif_list, DEFAULT_MIXTURE_MODEL, output_dir, P, Y, args.n_runs)
    print(f"  > MAPP Total Time: {mapp_res['timings']['total']:.2f}s")
    
    print("\nRunning ML-Pure + IAST Grid ...")
    iast_res = benchmark_ml_iast_grid(cif_list, DEFAULT_PURE_MODEL, output_dir, P, Y, args.n_runs)
    print(f"  > IAST Total Time: {iast_res['timings']['total']:.2f}s")
    
    # 4. Results
    plot_benchmark_results(mapp_res, iast_res, output_dir, len(cif_list), n_points)
    
    # Save CSV
    df = pd.DataFrame([{
        "Method": "MAPP",
        "TotalTime": mapp_res["timings"]["total"],
        "Analysis": "Model Load + Batch Inference (GPU)"
    }, {
        "Method": "ML-Pure+IAST",
        "TotalTime": iast_res["timings"]["total"],
        "Analysis": "Model Load + Pure Infer + Fitting + IAST Loop (CPU)"
    }])
    df.to_csv(output_dir / "results.csv", index=False)
    
    print(f"\nSpeedup MAPP/IAST: {iast_res['timings']['total']/mapp_res['timings']['total']:.2f}x")

if __name__ == "__main__":
    main()
