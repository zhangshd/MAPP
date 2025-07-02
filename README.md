# A Generic Model for Mixture Adsorption Property Prediction (MAPP): Toward Efficient Discovery of Metal-Organic Frameworks for CO2 Capture

## Overview

MAPP is a comprehensive machine learning framework for predicting gas adsorption properties in Metal-Organic Frameworks (MOFs) under realistic conditions (e.g., pressure and gas composition). The project combines Grand Canonical Monte Carlo (GCMC) simulations, advanced molecular representations, and deep learning models to predict adsorption isotherms for CO2 and N2 in MOF structures.

## Project Structure

```
MAPP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── pyrightconfig.json          # Python type checking configuration
│
├── CGCNN_MT/                   # Crystal Graph Convolutional Neural Network Multi-Task
│   ├── main.py                 # Main training script
│   ├── config.py               # Configuration settings
│   ├── predict.py              # Prediction script
│   ├── inference.py            # Inference script for new MOFs
│   ├── utils.py                # Utility functions
│   ├── hyperopt.py             # Hyperparameter optimization
│   ├── slurm_sub.py            # SLURM job submission
│   ├── datamodule/             # Data loading and processing
│   ├── module/                 # Model architectures
│   └── data/                   # Training data
│
├── MOFTransformer/             # MOFTransformer model implementation
│   ├── main.py                 # Main training script
│   ├── config.py               # Configuration settings
│   ├── predict.py              # Prediction script
│   ├── inference.py            # Inference script for new MOFs
│   ├── uncertainty.py          # Uncertainty quantification
│   ├── test.py                 # Model testing
│   ├── run.py                  # Training runner
│   ├── slurm_sub.py            # SLURM job submission
│   ├── datamodule/             # Data loading and processing
│   ├── module/                 # Model architectures
│   ├── models/                 # Pre-trained models
│   └── logs/                   # Training logs
│
├── GCMC/                       # Grand Canonical Monte Carlo simulations
│   ├── gcmc_process.py         # Main GCMC process management
│   ├── gcmc_template.py        # GCMC simulation templates
│   ├── gcmc_data_extract.py    # Data extraction from GCMC results
│   ├── utils.py                # GCMC utility functions
│   ├── molecule_utils.py       # Molecular manipulation utilities
│   ├── gcmc_task_*.py          # Task management scripts
│   ├── data/                   # GCMC simulation data
│   └── FF/                     # Force field parameters
│
├── descriptors/                # MOF descriptor generation
│   ├── feature_generation.py   # RAC and Zeo++ descriptor generation
│   ├── RAC_getter.py           # RAC descriptor calculation
│   ├── solvent_removal.py      # Solvent removal utilities
│   └── feature_folders/        # Generated descriptors
│
├── results/                    # Analysis results and comparisons
├── inference/                  # Inference results
├── logs/                       # Training and simulation logs
│
└── *.ipynb                     # Jupyter notebooks for workflow
```

## Features

### Core Capabilities
- **Multi-scale MOF Representation**: Combines crystal graph networks and 3D grid-based transformers
- **Multi-task Learning**: Simultaneous prediction of adsorption loadings and selectivities for CO2 and N2
- **GCMC Simulation Integration**: Automated generation of reference data using RASPA
- **Uncertainty Quantification**: Model uncertainty estimation using deep ensembles and LSV analysis
- **High-throughput Screening**: Efficient prediction for large MOF databases

### Supported Properties
- Gas adsorption isotherms (CO2, N2)
- Selectivity calculations
- Void fraction and porosity analysis

## Workflow

### 1. Data Preparation
- **CIF Processing**: Clean and validate MOF crystal structures
- **Graph Generation**: Create crystal graph representations
- **Grid Generation**: Generate 3D energy grids using GRIDAY
- **Descriptor Calculation**: Compute RAC and Zeo++ descriptors

### 2. GCMC Simulation
- **Task Creation**: Generate simulation input files
- **Parallel Execution**: Run high-throughput GCMC simulations
- **Data Extraction**: Parse simulation results and compile datasets

### 3. Model Training
- **CGCNN-MT**: Crystal graph-based multi-task learning
- **MOFTransformer**: Transformer-based model with graph and grid inputs
- **Multi-task Optimization**: Joint training on multiple properties

### 4. Prediction and Analysis
- **Inference**: Predict properties for new MOF structures
- **Uncertainty Estimation**: Quantify prediction reliability
- **Comparative Analysis**: Compare with experimental and simulation data

## Main Scripts Usage

### Data Preparation Scripts

#### MOF Structure Processing
```bash
# Prepare graph and grid data from CIF files
python MOFTransformer/datamodule/prepare_data.py \
    --cif_dir /path/to/cif/files \
    --saved_dir /path/to/output \
    --n_cpus 4
```

### GCMC Simulation Scripts

#### Task Creation and Execution
```bash
# Create GCMC simulation tasks
python GCMC/gcmc_create_task.py \
    --workdir /path/to/simulation/workspace \
    --cif_dir /path/to/cif/files \
    --mof_list mof_names.txt

# Submit simulation jobs
python GCMC/gcmc_task_submit.py \
    --workdir /path/to/simulation/workspace

# Extract simulation results
python GCMC/gcmc_data_extract.py \
    --workdir /path/to/simulation/workspace \
    --output_file results.csv
```

### Model Training Scripts

#### CGCNN Multi-Task Training
```bash
# Train CGCNN-MT model
python CGCNN_MT/main.py \
    --task_cfg ads_co2_n2 \
    --model_name cgcnn \
    --max_epochs 100 \
    --devices 1 \
    --per_gpu_batchsize 32
```

#### MOFTransformer Training
```bash
# Train MOFTransformer model
python MOFTransformer/main.py \
    --task_cfg ads_co2_n2 \
    --model_name extranformerv3 \
    --load_path models/pmtransformer.ckpt \
    --max_epochs 50 \
    --per_gpu_batchsize 16
```

### Prediction Scripts

#### CGCNN-MT Prediction
```bash
# Make predictions using trained CGCNN-MT model
python CGCNN_MT/predict.py \
    --model_dir /path/to/trained/model \
    --split test

# Inference on new MOFs
python CGCNN_MT/inference.py \
    --model_dir /path/to/trained/model \
    --cif_list new_mofs.txt \
    --output_dir predictions/
```

#### MOFTransformer Prediction
```bash
# Make predictions using trained MOFTransformer
python MOFTransformer/predict.py \
    --root_dataset /path/to/dataset \
    --load_path /path/to/model.ckpt \
    --split test \
    --save_dir predictions/

# Inference on new MOFs
python MOFTransformer/inference.py \
    --model_dir /path/to/trained/model \
    --cif_dir /path/to/new/cifs \
    --output_dir predictions/
```

### Analysis and Comparison Scripts

#### Model Performance Analysis
```bash
# Analyze model results and generate comparison plots
python -c "
import pandas as pd
from pathlib import Path
# Load and analyze prediction results
results_dir = Path('results/')
# Analysis code continues in Jupyter notebooks
"
```

## Configuration

### Model Configurations
- **CGCNN-MT**: Crystal graph convolution with multi-task heads
- **MOFTransformer**: Combines graph and 3D grid representations
- **Task Types**: Regression (adsorption, selectivity)

### GCMC Parameters
- **Temperature**: 298 K (customizable)
- **Pressure Range**: 0.0001 - 10 bar
- **Components**: CO2, N2
- **Force Fields**: UFF, TraPPE
- **Cycles**: 5000 (equilibration and production)

### Data Splits
- **Training**: 80% of MOF database
- **Validation**: 10% for hyperparameter tuning
- **Test**: 10% for final evaluation
- **Clustering-based splits** available for domain generalization

## Requirements

### Dependencies
- Python 3.8+
- PyTorch Lightning
- PyTorch Geometric
- MOFTransformer package
- RASPA simulation software
- Zeo++ analysis tools
- molSimplify (for RAC descriptors)

### Hardware Requirements
- GPU recommended for model training
- Multi-core CPU for GCMC simulations
- Sufficient storage for simulation data (TB scale)


## License

This project is licensed under the MIT License



**Note**: This is a research project. Please validate predictions against experimental data when possible and consider uncertainty estimates in decision-making applications.
