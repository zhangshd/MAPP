'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-12-02 10:25:45
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer)

from sacred import Experiment
import os
from pathlib import Path

ex = Experiment("mof_stability", save_git_info=False)


@ex.config
def cfg():
    # Basic Training Control
    batch_size = 8  # Batch size
    num_workers = 4  # Number of worker processes for data loading
    random_seed = 42  # Random seed
    accelerator = "gpu"  # Accelerator type
    devices = 1  # Number of devices
    max_epochs = 50  # Maximum number of training epochs
    limit_train_batches = None  # Limit on training batches
    limit_val_batches = None  # Limit on validation batches
    auto_lr_bs_find = False  # Auto learning rate and batch size finder flag
    progress_bar = True  # Progress bar display flag

    # Loss Function
    focal_alpha = 0.25  # Focal loss alpha parameter
    focal_gamma = 2  # Focal loss gamma parameter

    # Optimizer
    optim = 'adam'  # Optimizer type
    lr = 1e-3  # Learning rate
    weight_decay = 0.01  # Weight decay
    momentum = 0.9  # Momentum parameter
    optim_config = "fine"  # Optimizer configuration: coarse or fine
    group_lr = True  # Group learning rate flag
    lr_mult = 10  # Learning rate multiplier for multi-task learning heads
    

    # LR Scheduler
    lr_scheduler = 'reduce_on_plateau'  # Learning rate scheduler type: multi_step, cosine, reduce_on_plateau
    lr_decay_steps = 20  # Learning rate decay steps
    lr_milestones = [10, 20, 30, 50]  # Learning rate milestones
    lr_decay_rate = 0.8  # Learning rate decay rate
    lr_decay_min_lr = 1e-6  # Minimum learning rate for decay
    max_steps = -1  # Maximum number of training steps
    decay_power = (
        1  # Power of polynomial decay function
                   ) 
    warmup_steps = 0.05
    

    # Restart Control
    load_best = False  # Load best model flag
    load_dir = None  # Directory to load the model from
    load_ver = None  # Version of the model to load
    load_v_num = None  # Number of the model to load

    # Training Info
    log_dir = 'logs'  # Log directory
    patience = 10  # Patience
    min_delta = 0.001  # Minimum change
    monitor = 'val_Metric'  # Monitoring metric
    mode = 'max'  # Mode
    eval_freq = 10  # valset Evaluation frequency

    # Data Module Hyperparameters
    max_num_nbr = 10  # Maximum number of neighbors
    radius = 8  # Radius
    dmin = 0  # Minimum distance
    step = 0.2  # Step
    use_cell_params = False  # Use cell parameters flag
    use_extra_fea = True  # Use extra features flag
    task_weights = None
    
    
    # Model Hyperparameters
    model_name = 'cgcnn'  # Model name
    atom_fea_len = 128  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 256  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 4  # Number of hidden layers
    att_S = 64  # S parameter
    dropout_prob = 0.0  # Dropout probability
    att_pooling = False # Attention pooling flag
    task_norm = True  # Task normalization flag
    dwa_temp = 2.0  # DWA temperature parameter
    dwa_alpha = 0.8  # DWA alpha parameter
    noise_var = 0.1  # Noise variance parameter


@ex.named_config
def cgcnn():
    model_name = 'cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 1  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = False  # Atom layer normalization flag
    output_softplus = True  # Output softplus flag

@ex.named_config
def cgcnn_raw():
    model_name = 'cgcnn_raw'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag

@ex.named_config
def att_cgcnn():
    model_name = 'att_cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 1  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention

@ex.named_config
def cgcnn_langmuir():
    """CGCNN with Langmuir-gated heads for adsorption tasks."""
    model_name = 'cgcnn_langmuir'
    atom_fea_len = 64
    extra_fea_len = 128
    h_fea_len = 128
    n_conv = 3
    n_h = 1
    dropout_prob = 0.0
    use_extra_fea = True
    use_cell_params = False
    atom_layer_norm = False
    
    # Langmuir gating configuration
    langmuir_learnable_b = True
    langmuir_b_init = 1.0
    langmuir_softplus = True
    langmuir_power = 1.0
    langmuir_learnable_power = True
    langmuir_power_min = 1.0
    langmuir_power_max = 5.0
    langmuir_output_transform = "symlog"
    langmuir_symlog_threshold = 1e-4
    arcsinh_pressure_idx = 0
    co2_fraction_idx = 2

@ex.named_config
def cgcnn_uni_atom():
    model_name = 'cgcnn_uni_atom'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    max_graph_len = 300  # Maximum number of atoms in a graph
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 1  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention
    reconstruct = False  # Reconstruct atom features into fixed length gragph representation flag

@ex.named_config
def fcnn():
    model_name = 'fcnn'  # Model name
    extra_fea_len = 128  # Extra feature length
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability

@ex.named_config
def att_fcnn():
    model_name = 'att_fcnn'  # Model name
    extra_fea_len = 128  # Extra feature length
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention
    

@ex.named_config
def ads_co2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2']
    task_types = ['regression_log']
    log_press = True

@ex.named_config
def ads_n2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsN2']
    task_types = ['regression_log']
    log_press = True

@ex.named_config
def ads_co2_n2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2', 'AdsN2']
    task_types = ['regression_log', 'regression_log']
    log_press = True
    
@ex.named_config
def qst_co2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['QstCO2']
    task_types = ['regression']
    log_press = True

@ex.named_config
def qst_n2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['QstN2']
    task_types = ['regression']
    log_press = True


@ex.named_config
def ads_qst_co2_n2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2', 'AdsN2', 'QstCO2', 'QstN2']
    task_types = ['regression_log', 'regression_log', 'regression', 'regression']
    loss_aggregation = 'sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None
    log_press = True

@ex.named_config
def ads_s_qst_co2_n2():
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2', 'AdsN2', 'AdsS', 'QstCO2', 'QstN2']
    task_types = ['regression_log', 'regression_log', 'regression_log', 'regression', 'regression']
    loss_aggregation = 'sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None
    log_press = True

@ex.named_config
def ads_s_co2_n2():
    exp_name = "ads_s_co2_n2"
    data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['logAdsCO2', 'logAdsN2', 'logAdsS']
    task_types = ['regression', 'regression', 'regression']
    loss_aggregation = 'sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None
    log_press = True

@ex.named_config
def ads_qst_co2_n2_debug():
    data_dir = 'data/ddmof/mof_split_val10_test10_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2', 'AdsN2', 'QstCO2', 'QstN2']
    task_types = ['regression_log', 'regression_log', 'regression', 'regression']
    loss_aggregation = 'sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None
    log_press = True

@ex.named_config
def ads_qst_co2_n2_test():
    data_dir = 'data/ddmof/mof_split_val100_test100_seed0'  # Data directory
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 5e-3
    tasks = ['AdsCO2', 'AdsN2', 'QstCO2', 'QstN2']
    task_types = ['regression_log', 'regression_log', 'regression', 'regression']
    loss_aggregation = 'sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None
    log_press = True


# ============ Symlog Transformed Adsorption Configurations ============

@ex.named_config
def ads_symlog_co2_n2():
    """Symlog-transformed CO2/N2 mixture adsorption prediction"""
    exp_name = "ads_symlog_co2_n2"
    data_dir = 'data/ddmof/mof_split_val1000_test1000_seed0_org' # GMOF
    # data_dir = 'data/ddmof/mof_cluster_split_val1_test3_seed0_org'  # GCluster
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 1e-4
    loss_aggregation = 'sum'
    tasks = ['SymlogAbsLoadingCO2', 'SymlogAbsLoadingN2']
    task_types = ['regression', 'regression']
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    log_press = False  # the pressure column is already transformed
    symlog_threshold = 1e-4  # Symlog transformation threshold
    # Selectivity auxiliary loss
    selectivity_loss_weight = 0.1

@ex.named_config
def ads_symlog_co2_n2_pure():
    """Symlog-transformed pure component adsorption prediction"""
    exp_name = "ads_symlog_co2_n2_pure"
    data_dir = 'data/ddmof/mof_split_val1000_test1000_seed0_co2_n2_org'
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 1e-4
    loss_aggregation = 'sum'
    tasks = ['SymlogAbsLoadingCO2', 'SymlogAbsLoadingN2']
    task_types = ['regression', 'regression']
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]"]  # No CO2Fraction for pure components
    log_press = False
    symlog_threshold = 1e-4

@ex.named_config
def ads_symlog_qst_co2_n2():
    """Symlog-transformed adsorption + Qst prediction"""
    exp_name = "ads_symlog_qst_co2_n2"
    data_dir = 'data/ddmof/mof_split_val1000_test1000_seed0_org'
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 32
    lr = 1e-4
    loss_aggregation = 'sum'
    tasks = ['SymlogAbsLoadingCO2', 'SymlogAbsLoadingN2', 'QstCO2', 'QstN2']
    task_types = ['regression', 'regression', 'regression', 'regression']
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    log_press = False
    symlog_threshold = 1e-4
    selectivity_loss_weight = 0.0  # Optional: enable selectivity loss

@ex.named_config
def test_symlog():
    """
    Test config for Symlog-transformed adsorption prediction.
    Uses small dataset for quick validation.
    """
    exp_name = "test_symlog"
    data_dir = 'data/ddmof/mof_split_val10_test10_seed0_org'
    data_dir = str(Path(__file__).parent/data_dir)
    batch_size = 16
    max_epochs = 5
    lr = 1e-4
    loss_aggregation = 'sum'
    tasks = ['SymlogAbsLoadingCO2', 'SymlogAbsLoadingN2']
    task_types = ['regression', 'regression']
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    log_press = False
    use_extra_fea = True
    use_cell_params = False
    
    # Symlog configuration
    symlog_threshold = 1e-4
    
    # Selectivity auxiliary loss
    selectivity_loss_weight = 0.1
    
    # MAPE threshold
    mape_threshold = 0.01
