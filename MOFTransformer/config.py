# MOFTransformer version 2.1.3
import os
from sacred import Experiment
from moftransformer import __root_dir__
from moftransformer.utils.validation import _set_load_path
from pathlib import Path

ex = Experiment("pretrained_mof", save_git_info=False)


# def _loss_names(d):
#     ret = {
#         "ggm": 0,  # graph grid matching
#         "mpp": 0,  # masked patch prediction
#         "mtp": 0,  # mof topology prediction
#         "vfp": 0,  # (accessible) void fraction prediction
#         "moc": 0,  # metal organic classification
#         "bbc": 0,  # building block classification
#         "tsr": 0,  # thermal stability regression
#         "ssc": 0,  # solvent stability classification
#         "classification": 0,  # classification
#         "regression": 0,  # regression
#     }
#     ret.update(d)
#     return ret

@ex.config
def config():
    """
    # prepare_data
    max_num_atoms = 300
    min_length = 30
    max_length = 60
    radius = 8
    max_nbr_atoms = 12
    """

    # model
    exp_name = "pretrained_mof"
    model_name = "extranformerv1" # extranformerv2, extranformerv1
    seed = 42
    noise_var = 0.1
    limit_train_batches = None

    # graph seeting
    #max_supercell_atoms = None  # number of maximum atoms in supercell atoms
    atom_fea_len = 64
    nbr_fea_len = 64
    max_graph_len = 300  # number of maximum nodes in graph
    max_nbr_atoms = 12

    # grid setting
    img_size = 30
    patch_size = 5  # length of patch
    in_chans = 1  # channels of grid image
    max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
    draw_false_grid = False

    # extra features
    extra_norm = "batch"  # batch, layer, none

    # transformer setting
    hid_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    mpp_ratio = 0.15

    # downstream
    # downstream = ""
    # n_classes = 0

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
    max_epochs = 20
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 0  # multiply lr for pretrained model

    # PL Trainer Setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False
    progress_bar = False

    # callbacks
    patience = 10
    min_delta = 0.001

    # below params varies with the environment
    root_dataset = os.path.join(__root_dir__, "examples/dataset")
    log_dir = "logs/"
    batch_size = 64  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
    accelerator = "auto"
    devices = "auto"
    num_nodes = 1

    # load_path = _set_load_path('models/pmtransformer.ckpt')
    load_path = None

    num_workers = 2  # the number of cpu's core
    precision = "16-mixed"  #  "32-true", "16-mixed"

    # normalization target
    mean = None
    std = None

    # visualize
    visualize = False  # return attention map


@ex.named_config
def test():
    exp_name = "test"
    model_name = "extranformerv3"
    root_dataset = "data/ddmof/mof_split_val10_test10_seed0_org"
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
        # 'QstCO2': "regression", 
        # 'QstN2': "regression", 
    }
    max_epochs = 5
    per_gpu_batchsize = 16
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    extra_bins = 32

    selectivity_loss_weight = 0  # Weight for log-selectivity loss
    output_softplus=True

@ex.named_config
def test_org_v4():
    """
    Test config for ExTransformerV4 with Langmuir gating.
    Uses small dataset for quick validation.
    """
    exp_name = "test_org_v4"
    model_name = "extranformerv4"
    root_dataset = "data/ddmof/mof_split_val10_test10_seed0_org"
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
        # 'QstCO2': "regression", 
        # 'QstN2': "regression", 
    }
    max_epochs = 5
    per_gpu_batchsize = 16
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    extra_bins = 32
    
    # Langmuir gating configuration
    langmuir_learnable_b = True   # Whether b parameter is learnable
    langmuir_b_init = 1.0         # Initial value for b (1/bar)
    langmuir_softplus = True      # Use softplus for non-negative output
    arcsinh_pressure_idx = 0      # Index of ArcsinhPressure in extra_fea
    co2_fraction_idx = 2          # Index of CO2Fraction in extra_fea
    
    # Selectivity auxiliary loss configuration
    selectivity_loss_weight = 0.1  # Weight for log-selectivity loss

    langmuir_learnable_power = True   # enabel learnable power for pressure in langmuir gate (P^n/(1+bP^n))
    langmuir_power = 1.0              # initial value for power
    langmuir_power_min = 1.0          # minimum value for power
    langmuir_power_max = 5.0          # maximum value for power
    langmuir_output_transform = "symlog"  # Output transform to match label scale
    langmuir_symlog_threshold = 1e-4      # Symlog threshold (must match preprocessing)

@ex.named_config
def ads_qst_co2_n2():
    exp_name = "ads_qst_co2_n2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
        'QstCO2': "regression", 
        'QstN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32
    
@ex.named_config
def ads_co2_n2():
    exp_name = "ads_co2_n2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_co2():
    exp_name = "ads_co2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_n2():
    exp_name = "ads_n2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_s_qst_co2_n2():
    exp_name = "ads_s_qst_co2_n2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
        'logAdsS': "regression",
        'QstCO2': "regression", 
        'QstN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_s_co2_n2():
    exp_name = "ads_s_co2_n2"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
        'logAdsS': "regression",
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_s_co2_n2_abs():
    exp_name = "ads_s_co2_n2_abs"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'LogAbsLoadingCO2': "regression", 
        'LogAbsLoadingN2': "regression", 
        'LogAbsLoadingS': "regression",
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "LogPressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_co2_n2_org():
    exp_name = "ads_co2_n2_org"
    # root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    root_dataset = 'data/ddmof/mof_cluster_split_val1_test3_seed0_org' # GCluster
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    extra_bins=32
    selectivity_loss_weight = 0.0  # Weight for log-selectivity loss
    output_softplus=True
    

@ex.named_config
def ads_co2_n2_org_v4():
    """
    ExTransformerV4 with Langmuir gating for CO2/N2 adsorption prediction.
    Uses Symlog-transformed pressure and partial pressure calculation.
    Ensures thermodynamic consistency: q(P=0)=0 and saturation at high P.
    """
    exp_name = "ads_co2_n2_org_v4"
    model_name = "extranformerv4"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    # root_dataset = 'data/ddmof/mof_cluster_split_val1_test3_seed0_org' # GCluster
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    extra_bins = 32
    
    # Langmuir gating configuration
    langmuir_learnable_b = True   # Whether b parameter is learnable
    langmuir_b_init = 1.0         # Initial value for b (1/bar)
    langmuir_softplus = True      # Use softplus for non-negative output
    arcsinh_pressure_idx = 0      # Index of ArcsinhPressure in extra_fea
    co2_fraction_idx = 2          # Index of CO2Fraction in extra_fea
    langmuir_learnable_power = True   # enabel learnable power for pressure in langmuir gate (P^n/(1+bP^n))
    langmuir_power = 1.0              # initial value for power
    langmuir_power_min = 1.0          # minimum value for power
    langmuir_output_transform = "symlog"  # Output transform to match label scale
    langmuir_symlog_threshold = 1e-4      # Symlog threshold

    selectivity_loss_weight = 0.1  # Weight for log-selectivity loss

@ex.named_config
def ads_co2_n2_pure_v4():
    """
    ExTransformerV4 with Langmuir gating for pure-component CO2/N2 adsorption.
    Uses symlog-transformed pressure. For pure components, CO2Fraction is
    implicitly 1.0 for CO2 task and 0.0 for N2 task.
    """
    exp_name = "ads_co2_n2_pure_v4"
    model_name = "extranformerv4"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_co2_n2_org'  # Pure component data
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]"]  # No CO2Fraction for pure components
    extra_bins = 32
    
    # Langmuir gating configuration
    langmuir_learnable_b = True   # Whether b parameter is learnable
    langmuir_b_init = 1.0         # Initial value for b (1/bar)
    langmuir_softplus = True      # Use softplus for non-negative output
    arcsinh_pressure_idx = 0      # Index of ArcsinhPressure in extra_fea
    langmuir_learnable_power = True   # enabel learnable power for pressure in langmuir gate (P^n/(1+bP^n))
    langmuir_power = 1.0              # initial value for power
    langmuir_power_min = 1.0          # minimum value for power
    langmuir_output_transform = "symlog"  # Output transform to match label scale
    langmuir_symlog_threshold = 1e-4      # Symlog threshold
    # Note: co2_fraction_idx not needed for pure component; 
    # LangmuirGatedRegressionHead will use implicit fraction (1.0 for CO2, 0.0 for N2)


@ex.named_config
def ads_qst_co2_n2_org_v4():
    """
    ExTransformerV4 with Langmuir gating for CO2/N2 adsorption prediction.
    Uses arcsinh-transformed pressure and partial pressure calculation.
    Ensures thermodynamic consistency: q(P=0)=0 and saturation at high P.
    """
    exp_name = "ads_qst_co2_n2_org_v4"
    model_name = "extranformerv4"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'ArcsinhAbsLoadingCO2': "regression", 
        'ArcsinhAbsLoadingN2': "regression", 
        'QstCO2': "regression", 
        'QstN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["ArcsinhPressure[bar]", "LogPressure[bar]", "CO2Fraction"]
    extra_bins = 32
    
    # Langmuir gating configuration
    langmuir_learnable_b = True   # Whether b parameter is learnable
    langmuir_b_init = 1.0         # Initial value for b (1/bar)
    langmuir_softplus = True      # Use softplus for non-negative output
    arcsinh_pressure_idx = 0      # Index of ArcsinhPressure in extra_fea
    co2_fraction_idx = 2          # Index of CO2Fraction in extra_fea

@ex.named_config
def ads_qst_co2_n2_org_v4_sel():
    """
    ExTransformerV4 with Langmuir gating and selectivity auxiliary loss.
    Same as ads_qst_co2_n2_org_v4 but with log-selectivity loss for 
    additional physical constraint on CO2/N2 relative adsorption.
    """
    exp_name = "ads_qst_co2_n2_org_v4_sel"
    model_name = "extranformerv4"
    # root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # GMOF
    root_dataset = 'data/ddmof/mof_cluster_split_val1_test3_seed0_org' # GCluster
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'SymlogAbsLoadingCO2': "regression", 
        'SymlogAbsLoadingN2': "regression", 
        'QstCO2': "regression", 
        'QstN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = False
    use_extra_fea = True
    use_cell_params = False
    condi_cols = ["ArcsinhPressure[bar]", "SymlogPressure[bar]", "CO2Fraction"]
    extra_bins = 32
    
    # Langmuir gating configuration
    langmuir_learnable_b = True
    langmuir_b_init = 1.0
    langmuir_softplus = True
    arcsinh_pressure_idx = 0
    co2_fraction_idx = 2
    
    # Selectivity auxiliary loss configuration
    selectivity_loss_weight = 0.1  # Weight for log-selectivity loss

    langmuir_learnable_power = True   # enabel learnable power for pressure in langmuir gate (P^n/(1+bP^n))
    langmuir_power = 1.0              # initial value for power
    langmuir_power_min = 1.0          # minimum value for power
    langmuir_power_max = 5.0          # maximum value for power
    langmuir_output_transform = "symlog"  # Output transform to match label scale
    langmuir_symlog_threshold = 1e-4      # Symlog threshold

@ex.named_config
def ads_s_co2_n2_org():
    exp_name = "ads_s_co2_n2_org"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'ArcsinhAbsLoadingCO2': "regression", 
        'ArcsinhAbsLoadingN2': "regression", 
        'ArcsinhAbsLoadingS': "regression",
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

@ex.named_config
def ads_s_qst_co2_n2_org():
    exp_name = "ads_s_qst_co2_n2_org"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_org'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'AbsLoadingCO2': "regression_log", 
        'AbsLoadingN2': "regression_log", 
        'AbsLoadingS': "regression_log",
        'QstCO2': "regression", 
        'QstN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32


@ex.named_config
def ads_co2_pure():
    exp_name = "ads_co2_pure"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_co2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]"]
    extra_bins=32

@ex.named_config
def ads_n2_pure():
    exp_name = "ads_n2_pure"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_n2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]"]
    extra_bins=32

@ex.named_config
def ads_co2_n2_pure():
    exp_name = "ads_co2_n2_pure"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_co2_n2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]"]
    extra_bins=32

@ex.named_config
def ads_s_co2_n2_mix():
    exp_name = "ads_s_co2_n2_mix"
    root_dataset = 'data/ddmof/mof_split_val1000_test1000_seed0_mixture'  # Data directory
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
        'logAdsS': "regression",
    }
    max_epochs = 50
    per_gpu_batchsize = 32
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32