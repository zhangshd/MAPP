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
    learning_rate = 1e-6
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
    max_epochs = 20
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 100  # multiply lr for downstream heads

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
    tasks = {
        'logAdsCO2': "regression", 
        'logAdsN2': "regression", 
        # 'QstCO2': "regression", 
        # 'QstN2': "regression", 
    }
    root_dataset = "data/ddmof/mof_split_val10_test10_seed0"
    root_dataset = str(Path(__file__).parent.parent/"CGCNN_MT"/root_dataset)
    max_epochs = 2
    # batch_size = 16
    per_gpu_batchsize = 16
    log_press = True
    use_extra_fea = True  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    condi_cols = ["Pressure[bar]", "CO2Fraction"]
    extra_bins=32

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