# A Generic Model for Mixture Adsorption Property Prediction (MAPP): Toward Efficient Discovery of Metal-Organic Frameworks for CO2 Capture

## Overview

MAPP is a machine learning framework for predicting gas adsorption properties in Metal-Organic Frameworks (MOFs) under various conditions (e.g., pressure and gas composition). The project combines Grand Canonical Monte Carlo (GCMC) simulations, advanced molecular representations, and deep learning models to predict adsorption isotherms for CO2 and N2 in MOF structures.

## Project Structure

```
MAPP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── pyrightconfig.json          # Python type checking configuration
│
├── CGCNN_MT/                   # Crystal Graph Convolutional Neural Network Multi-Task
│   ├── main.py                 # Main training script
│   ├── config.py               # Configuration settings (task configs: ads_symlog_co2_n2, etc.)
│   ├── predict.py              # Prediction on test set
│   ├── inference.py            # Inference script for new MOFs
│   ├── utils.py                # Utility functions
│   ├── hyperopt.py             # Hyperparameter optimization
│   ├── slurm_sub.py            # SLURM job submission
│   ├── datamodule/             # Data loading and processing
│   ├── module/                 # Model architectures (cgcnn, cgcnn_langmuir, att_cgcnn)
│   └── data/                   # Training data (available at https://zenodo.org/records/15796756)
│
├── MOFTransformer/             # MOFTransformer model implementation
│   ├── main.py                 # Main training script
│   ├── config.py               # Configuration settings (task configs: ads_co2_n2_org_v4, etc.)
│   ├── predict.py              # Prediction on test set
│   ├── inference.py            # Inference script for new MOFs
│   ├── uncertainty.py          # Uncertainty quantification
│   ├── test.py                 # Model testing
│   ├── run.py                  # Training runner
│   ├── slurm_sub.py            # SLURM job submission
│   ├── datamodule/             # Data loading and processing
│   ├── module/                 # Model architectures (ExTransformerV3, ExTransformerV4)
│   ├── models/                 # Pre-trained models (pmtransformer.ckpt)
│   └── logs/                   # Training logs (available at https://zenodo.org/records/15796756)
│
├── RASPATOOLS/                 # GCMC toolkit (see RASPATOOLS/README.md)
│
├── descriptors/                # MOF descriptor generation
│   ├── feature_generation.py   # RAC and Zeo++ descriptor generation
│   ├── RAC_getter.py           # RAC descriptor calculation
│   ├── solvent_removal.py      # Solvent removal utilities
│   └── feature_folders/        # Generated descriptors
│
├── experiments/                # Analysis and experiment scripts
│   ├── config.yaml             # Model and data path configuration
│   ├── 00_plot_raspa_validation.py   # RASPA validation plotting
│   ├── 01_data_analysis.py           # Data exploration
│   ├── 02_mof_clustering.py          # MOF clustering for data split
│   ├── 03_make_training_data.py      # Generate training datasets
│   ├── 04_model_results_analysis.py  # Model performance analysis
│   ├── 05_error_statistics.py        # Error distribution statistics
│   ├── 06_uncertainty_analysis.py    # Uncertainty quantification
│   ├── 07_compare_mapp_iast_methods.py  # Compare MAPP vs IAST
│   ├── 08_compare_wc_selectivity.py     # Working capacity analysis
│   ├── 09_benchmark_mapp_vs_iast.py     # MAPP vs IAST benchmark
│   └── 10_compare_experimental_mofs.py  # Experimental MOF validation
│
├── applications/               # User-friendly prediction scripts
│   ├── predict.py              # Main MAPP prediction script
│   ├── utils_iast.py           # IAST utilities (isotherm fitting, IAST prediction)
│   ├── demo_cifs/              # Demo CIF files for testing
│   └── output/                 # Prediction output directory
│
├── results/                    # Analysis results and comparisons
└── compress_data_and_models.py # Zenodo data packaging script
```

## Features

### Core Capabilities
- **Multi-scale MOF Representation**: Combines crystal graph networks and 3D grid-based transformers
- **Multi-task Learning**: Simultaneous prediction of adsorption loadings for CO2 and N2
- **Langmuir-gated Output**: Physically-constrained predictions ensuring q(P=0)=0 and saturation at high P
- **GCMC Simulation Integration**: Automated generation of reference data using RASPA
- **Uncertainty Quantification**: Model uncertainty estimation using deep ensembles
- **High-throughput Screening**: Efficient prediction for large MOF databases

### Supported Properties
- Gas adsorption isotherms (CO2, N2) - pure and mixture components
- Working capacity calculations
- Selectivity calculations

## Configuration

### Centralized Configuration (`experiments/config.yaml`)
```yaml
models:
  MAPP_GMOF: "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/version_14"
  MAPP_GCluster: "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/version_12"
  CGCNN_GMOF: "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_7"
  CGCNN_GCluster: "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_8"
  MAPPPure: "MOFTransformer/logs/ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer/version_2"

data:
  ddmof_cifs: "CGCNN_MT/data/ddmof/cifs"
  ddmof_data: "CGCNN_MT/data/ddmof"
```

### Data Transformations
MAPP uses **symlog transformation** for adsorption data:
```
symlog(x, threshold=1e-4) = sign(x) * log10(1 + |x|/threshold)
```
This preserves precision for small loading values while compressing large values.

### Data Splits
- **GMOF (MOF-based)**: Random split by MOF names (val=1000, test=1000)
- **GCluster (Cluster-based)**: Split by structural similarity clusters

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
- **CGCNN-MT**: Crystal graph-based multi-task learning with Langmuir gating
- **MOFTransformer (ExTransformerV4)**: Transformer-based model with graph, grid inputs and Langmuir gating
- **Multi-task Optimization**: Joint training on CO2 and N2 adsorption

### 4. Prediction and Analysis
- **Inference**: Predict properties for new MOF structures
- **Uncertainty Estimation**: Quantify prediction reliability
- **Comparative Analysis**: Compare with experimental and simulation data

## Applications

The `applications/` directory provides user-friendly scripts for quick predictions.

### Quick Prediction

```bash
# Basic prediction (working capacity and selectivity)
conda run -n mofnn python applications/predict.py --cif_dir YOUR_CIF_DIR

# With IAST comparison
conda run -n mofnn python applications/predict.py --cif_dir YOUR_CIF_DIR --use_iast
```

### Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cif_dir` | (required) | Directory with CIF files or a single CIF file |
| `--ads_pressure` | 1.0 | Adsorption pressure (bar) |
| `--ads_co2_frac` | 0.15 | Adsorption CO2 fraction |
| `--des_pressure` | 0.01 | Desorption pressure (bar) |
| `--des_co2_frac` | 0.9 | Desorption CO2 fraction |
| `--use_iast` | False | Enable IAST comparison |
| `--output_dir` | `applications/output` | Output directory |

### Output Files

| File | Description |
|------|-------------|
| `mapp_raw_predictions.csv` | Raw adsorption amounts at all conditions |
| `mapp_summary.csv` | Working capacity (WC) and selectivity (S) summary |
| `isotherm_models.json` | (IAST mode) Fitted isotherm model parameters |

## Main Scripts Usage

### Data Preparation Scripts

#### Generate Training Data from GCMC Results
```bash
python experiments/03_make_training_data.py \
    --data_dir CGCNN_MT/data/ddmof \
    --split_type mof \
    --val_size 1000 \
    --test_size 1000
```

#### MOF Structure Processing
```bash
# Prepare graph and grid data from CIF files
python MOFTransformer/datamodule/prepare_data.py \
    --cif_dir /path/to/cif/files \
    --saved_dir /path/to/output \
    --n_cpus 4
```

### Model Training Scripts

#### MOFTransformer Training (ExTransformerV4)
```bash
# Train mixture adsorption model with Langmuir gating
python MOFTransformer/main.py \
    --task_cfg ads_co2_n2_org_v4 \
    --load_path models/pmtransformer.ckpt \
    --max_epochs 50 \
    --per_gpu_batchsize 16

# Train pure component model
python MOFTransformer/main.py \
    --task_cfg ads_co2_n2_pure_v4 \
    --load_path models/pmtransformer.ckpt
```

#### CGCNN Multi-Task Training
```bash
# Train CGCNN-MT model with Langmuir gating
python CGCNN_MT/main.py \
    --task_cfg ads_symlog_co2_n2 \
    --model_name cgcnn_langmuir \
    --max_epochs 100 \
    --devices 1 \
    --per_gpu_batchsize 32
```

### Prediction Scripts

#### CGCNN-MT Prediction
```bash
# Make predictions on test set
python CGCNN_MT/predict.py \
    --model_dir logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_7

# Inference on new MOFs
python CGCNN_MT/inference.py
```

#### MOFTransformer Prediction
```bash
# Make predictions on test set
python MOFTransformer/predict.py \
    --root_dataset /path/to/dataset \
    --load_path /path/to/model.ckpt \
    --split test \
    --save_dir predictions/

# Inference on new MOFs
python MOFTransformer/inference.py
```

### Analysis Scripts

```bash
# Model performance analysis
python experiments/04_model_results_analysis.py

# Error statistics and visualization
python experiments/05_error_statistics.py

# Compare with experimental MOF data
python experiments/10_compare_experimental_mofs.py

# Benchmark MAPP vs IAST methods
python experiments/09_benchmark_mapp_vs_iast.py
```

## Model Configurations

### Task Configurations (MOFTransformer)
| Config Name | Description |
|-------------|-------------|
| `ads_co2_n2_org_v4` | Mixture CO2/N2 adsorption (ExTransformerV4 with Langmuir) |
| `ads_co2_n2_pure_v4` | Pure component adsorption (ExTransformerV4 with Langmuir) |
| `ads_qst_co2_n2_org_v4` | Adsorption + Qst prediction |

### Task Configurations (CGCNN-MT)
| Config Name | Description |
|-------------|-------------|
| `ads_symlog_co2_n2` | Symlog-transformed CO2/N2 mixture adsorption |
| `ads_symlog_co2_n2_pure` | Symlog-transformed pure component adsorption |
| `cgcnn_langmuir` | CGCNN with Langmuir-gated output heads |

### GCMC Parameters
- **Temperature**: 298 K (customizable)
- **Pressure Range**: 0.0001 - 10 bar
- **Components**: CO2, N2
- **Force Fields**: UFF, TraPPE
- **Cycles**: 5000 (equilibration and production)

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
