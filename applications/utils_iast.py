"""
IAST Utilities for MAPP Application
Provides isotherm fitting and IAST prediction functions.
Adapted from MOF-HTS/src/adsorption_analysis

Author: zhangshd
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import pygaps
import pygaps.modelling
import pygaps.parsing as pgp
from pygaps.iast import iast_point_fraction

logger = logging.getLogger(__name__)

# Default isotherm models to try (in order of preference)
# Note: Some model is not compatible with IAST, for more information, please refer to the pygaps documentation
DEFAULT_MODELS = ['Henry', 'DSLangmuir', 'Langmuir', 'Toth']


def fit_isotherm(
    pressures: np.ndarray,
    loadings: np.ndarray,
    mof_name: str,
    gas_name: str,
    temperature: float = 298.0,
    models_to_try: List[str] = None
) -> Optional[Dict]:
    """
    Fit isotherm models to pure component adsorption data.
    
    Args:
        pressures: Pressure array in bar
        loadings: Loading array in mol/kg
        mof_name: Name of the MOF
        gas_name: Name of the gas (CO2 or N2)
        temperature: Temperature in Kelvin
        models_to_try: List of model names to try
        
    Returns:
        Dictionary with fitted model info, or None if all fits failed
    """
    if models_to_try is None:
        models_to_try = DEFAULT_MODELS
    
    # Validate data
    if len(pressures) < 3 or len(loadings) < 3:
        logger.warning(f"Insufficient data points for {mof_name}-{gas_name}")
        return None
    
    if np.any(loadings < 0):
        logger.warning(f"Negative loadings found for {mof_name}-{gas_name}")
        loadings = np.maximum(loadings, 0)
    
    try:
        # Create pygaps PointIsotherm (mol/kg = mmol/g)
        isotherm = pygaps.PointIsotherm(
            pressure=pressures.tolist(),
            loading=loadings.tolist(),
            material=mof_name,
            adsorbate=gas_name,
            temperature=temperature,
            pressure_mode='absolute',
            pressure_unit='bar',
            loading_basis='molar',
            loading_unit='mmol',
            material_basis='mass',
            material_unit='g',
            temperature_unit='K'
        )
        
        # Try fitting models
        best_result = None
        best_r2 = -np.inf
        
        for model_name in models_to_try:
            try:
                fitted = pygaps.modelling.model_iso(isotherm, model=model_name, verbose=False)
                
                # Calculate R²
                pred_loadings = np.array([fitted.loading_at(p) for p in pressures])
                ss_res = np.sum((loadings - pred_loadings) ** 2)
                ss_tot = np.sum((loadings - np.mean(loadings)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if r2 > best_r2:
                    best_r2 = r2
                    fitted_json = pgp.isotherm_to_json(fitted)
                    best_result = {
                        'fitted_isotherm': json.loads(fitted_json),
                        'model_name': model_name,
                        'R2': float(r2),
                        'pressure_range': [float(pressures.min()), float(pressures.max())],
                        'loading_range': [float(loadings.min()), float(loadings.max())]
                    }
                    
            except Exception as e:
                logger.debug(f"Model {model_name} failed for {mof_name}-{gas_name}: {e}")
                continue
        
        if best_result:
            logger.info(f"Best fit for {mof_name}-{gas_name}: {best_result['model_name']} (R²={best_r2:.4f})")
        return best_result
        
    except Exception as e:
        logger.error(f"Isotherm fitting failed for {mof_name}-{gas_name}: {e}")
        return None


def load_model_isotherm(model_dict: Dict) -> pygaps.ModelIsotherm:
    """Load pygaps ModelIsotherm from dictionary."""
    fitted_json = json.dumps(model_dict['fitted_isotherm'])
    return pgp.isotherm_from_json(fitted_json)


def iast_predict(
    model_co2: Dict,
    model_n2: Dict,
    total_pressure: float,
    co2_fraction: float,
    temperature: float = 298.0
) -> Tuple[float, float]:
    """
    Predict mixture adsorption using IAST.
    
    Args:
        model_co2: Fitted CO2 isotherm model dict
        model_n2: Fitted N2 isotherm model dict
        total_pressure: Total pressure in bar
        co2_fraction: CO2 mole fraction in gas phase
        
    Returns:
        Tuple of (q_co2, q_n2) in mol/kg
    """
    try:
        iso_co2 = load_model_isotherm(model_co2)
        iso_n2 = load_model_isotherm(model_n2)
        
        mole_fractions = [co2_fraction, 1 - co2_fraction]
        
        result = iast_point_fraction(
            isotherms=[iso_co2, iso_n2],
            gas_mole_fraction=mole_fractions,
            total_pressure=total_pressure,
            warningoff=True
        )
        
        return float(result[0]), float(result[1])
        
    except Exception as e:
        logger.error(f"IAST prediction failed: {e}")
        return np.nan, np.nan


def save_models_json(models: Dict, output_file: Path) -> None:
    """Save fitted models to JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(models, f, indent=2)
    
    logger.info(f"Saved isotherm models to {output_file}")


def load_models_json(input_file: Path) -> Dict:
    """Load fitted models from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)
