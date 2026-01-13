'''
Author: zhangshd
Date: 2024-08-19 15:59:37
LastEditors: zhangshd
LastEditTime: 2024-12-02 17:20:24
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CGCNN_MT.datamodule.dataset import AtomCustomJSONInitializer, GaussianDistance
from CGCNN_MT.utils import load_model_from_dir
from CGCNN_MT.datamodule.prepare_data import make_prepared_data
from CGCNN_MT.datamodule.clean_cif import clean_cif
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import matplotlib
import pickle
import os, sys
import functools
import inspect
import shutil
matplotlib.use('Agg')



def process_cif(cif, saved_dir, clean=True, **kwargs):

    if isinstance(cif, str):
        cif = Path(cif)
    saved_dir = Path(saved_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(str(saved_dir/__name__))

    graphdata_dir = saved_dir / "graphdata"

    cif_id: str = cif.stem
    graphdata_dir.mkdir(exist_ok=True, parents=True)
    clean_cif_file = graphdata_dir / f"{cif_id}.cif"
    p_graphdata = graphdata_dir / f"{cif_id}.graphdata"
    if not clean_cif_file.exists() and clean:
        flag = clean_cif(cif, clean_cif_file)
        if not flag:
            return None
    else:
        shutil.copy(cif, clean_cif_file)
    if not p_graphdata.exists():
        p_graphdata = make_prepared_data(clean_cif_file, graphdata_dir, logger, **kwargs)
    return p_graphdata
    

class InferenceDataset(Dataset):
    def __init__(self, cif_list, co2frac, press, **kwargs):
        """
        Args:
            cif_list (list or str): list of cif file paths or a single cif file path.
        """
        if isinstance(cif_list, (str, Path)):
            self.cif_list = [Path(cif_list)]
        else:
            self.cif_list = [Path(cif) for cif in cif_list]

        self.split = "infer"
        self.radius = kwargs.get("radius", 8)
        self.max_num_nbr = kwargs.get("max_num_nbr", 12)
        self.dmin = kwargs.get("dmin", 0)
        self.step = kwargs.get("step", 0.2)
        self.use_cell_params = kwargs.get("use_cell_params", True)
        self.use_extra_fea = kwargs.get("use_extra_fea", True)
        self.task_id = kwargs.get("task_id", 0)
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.cif_ids = [cif.stem for cif in self.cif_list]
        self.log_press = kwargs.get("log_press", True)
        self.g_data ={}
        self.saved_dir = kwargs.get("saved_dir", Path(os.getcwd())/"inference")
        self.clean = kwargs.get("clean", True)
        
        atom_prop_json = Path(inspect.getfile(AtomCustomJSONInitializer)).parent/'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)

        if isinstance(co2frac, (float, int)):
            self.co2frac = [co2frac]
        else:
            self.co2frac = co2frac
        if isinstance(press, (float, int)):
            self.press = [press]
        else:
            self.press = press

        self.condi_cols = kwargs.get("condi_cols", [])
        self.symlog_threshold = kwargs.get("symlog_threshold", 1e-4)
        
        # Detect pressure format from condi_cols
        self.use_arcsinh = len(self.condi_cols) > 0 and "Arcsinh" in self.condi_cols[0]
        self.has_co2frac = len(self.condi_cols) > 1 and any("CO2Fraction" in col for col in self.condi_cols)

        ## inputs is a list of combinations of cif_ids, conditions
        inputs = []
        use_symlog_press = len(self.condi_cols) > 1 and "Symlog" in self.condi_cols[1]
        
        for file in self.cif_list:
            for press in self.press:
                if self.use_arcsinh:
                    # Format: [ArcsinhPressure, SymlogPressure/LogPressure, CO2Fraction]
                    arcsinh_press = np.arcsinh(press)
                    if use_symlog_press:
                        second_press = np.sign(press) * np.log10(1 + np.abs(press) / self.symlog_threshold)
                    else:
                        second_press = np.log10(press + 1e-5)
                    if self.has_co2frac:
                        for frac in self.co2frac:
                            inputs.append([file.stem, arcsinh_press, second_press, frac])
                    else:
                        inputs.append([file.stem, arcsinh_press, second_press])
                else:
                    # Legacy log format: [LogPressure, CO2Fraction]
                    if self.log_press:
                        log_press = np.log10(press + 1e-5)
                    else:
                        log_press = press
                    for frac in self.co2frac:
                        inputs.append([file.stem, log_press, frac])
        self.inputs = inputs
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.g_data.update(new_data.g_data)

    def setup(self, stage=None):

        for cif in self.cif_list:
            graphdata_file = process_cif(cif, self.saved_dir, clean=self.clean, 
                                         max_num_nbr=self.max_num_nbr, 
                                         radius=self.radius)
            if graphdata_file:
                self.g_data[cif.stem] = graphdata_file
            else:
                self.cif_ids.remove(cif.stem)
                self.inputs = [inp for inp in self.inputs if inp[0]!= cif.stem]
                print(f"Error: {cif} has been removed from the dataset due to errors during data preparation.")
        
    def __len__(self):
        return len(self.inputs)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        cif_id = self.inputs[idx][0]
        # print(cif_id, self.g_data[cif_id])
        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = data
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 10.0, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 10.0 for file: {self.g_data[cif_id]}"

        extra_fea = torch.FloatTensor(self.inputs[idx][1:])

        atom_fea = np.vstack([self.ari.get_atom_fea(i) for i in atom_num])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()
        assert isinstance(nbr_fea, torch.Tensor)

        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret_dict = {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "extra_fea": extra_fea,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        return dict_batch

def inference(cif_list, co2frac, press, model_dir, saved_dir, **kwargs):
    """
    Args:    
        cif_list (list or str): list of cif file paths or a single cif file path.
        model (MInterface): trained model.
    """

    # set up model
    model, trainer = load_model_from_dir(model_dir)
    clean = kwargs.get("clean", True)
    # set up dataset
    infer_dataset = InferenceDataset(cif_list, co2frac, press, saved_dir=saved_dir, clean=clean, **model.hparams)
    infer_dataset.setup()
    infer_loader = DataLoader(infer_dataset, 
                              batch_size=min(len(infer_dataset), model.hparams.get("batch_size", 8)), 
                            #   batch_size = 2,
                              shuffle=False, 
                              num_workers=model.hparams.get("num_workers", 2),
                              collate_fn=infer_dataset.collate
                              )

    outputs = trainer.predict(model, infer_loader)
    all_outputs = {}
    all_outputs[f"CifId"] = [d["cif_id"] for d in infer_dataset]
    
    # Pressure recovery based on condi_cols
    condi_cols = model.hparams.get("condi_cols", [])
    if len(condi_cols) > 0 and "Arcsinh" in condi_cols[0]:
        # Arcsinh format: P = sinh(arcsinh_P)
        all_outputs["Pressure[bar]"] = [np.sinh(d["extra_fea"][0].item()) for d in infer_dataset]
    else:
        # Legacy log format: P = 10^(log_P) - eps
        all_outputs["Pressure[bar]"] = [10**(d["extra_fea"][0].item()) - 1e-5 for d in infer_dataset]
    
    # CO2Fraction recovery
    if len(condi_cols) > 2 and "CO2Fraction" in condi_cols[2]:
        co2_frac_idx = 2
    elif len(condi_cols) > 1 and "CO2Fraction" in condi_cols[1]:
        co2_frac_idx = 1
    else:
        co2_frac_idx = 1  # Default
    
    if infer_dataset.has_co2frac:
        all_outputs["CO2Fraction"] = [d["extra_fea"][co2_frac_idx].item() for d in infer_dataset]
    else:
        all_outputs["CO2Fraction"] = [1.0 if "CO2" in model.hparams.get("tasks", [])[0] else 0.0 for d in infer_dataset]
    
    for task in model.hparams.get("tasks"):
        task_id = model.hparams["tasks"].index(task)
        task_tp = model.hparams["task_types"][task_id]
        all_outputs[f"{task}Predicted"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
        all_outputs[f'{task}_last_layer_fea'] = torch.cat([d[f'{task}_last_layer_fea'] for d in outputs], dim=0).cpu().numpy().squeeze()
        if "classification" in task_tp:
            all_outputs[f"{task}Prob"] = torch.cat([d[f"{task}_prob"] for d in outputs], dim=0).cpu().numpy()
    
    return all_outputs

if __name__ == "__main__":

#     cif_list = [
#         'ddmof_1009',
#         'ddmof_20281',
#         'ddmof_22102',
#         'ddmof_2497',
#         'ddmof_2707',
#         'ddmof_3777',
#         'ddmof_5422',
#         'ddmof_5714',
#         'ddmof_5980',
#         'ddmof_6457',
#         'ddmof_6466',
#         'ddmof_7238',
#         'ddmof_7267',
#         'ddmof_7277',
#         'ddmof_729',
#         'ddmof_8056',
#         'ddmof_8269',
#         'ddmof_9614',
#         'ddmof_9656',
#         'ddmof_9975'
#  ]
    clean = True
    cif_dir = Path(__file__).parent.parent/"GCMC/data/ddmof/cifs"
    # cif_dir = Path(__file__).parent.parent/"GCMC/data/CoREMOF2019/cifs"
    notes = cif_dir.parent.name if clean else cif_dir.parent.name + "_raw"
    # cif_list = [cif_dir/(cif + ".cif") for cif in cif_list]
    cif_list = list(cif_dir.glob("*.cif"))
    print("Number of cifs:", len(cif_list))
    
    # model_dir = Path(__file__).parent/"logs/AdsCO2_AdsN2_QstCO2_QstN2_seed42_cgcnn/version_2"
    # model_dir = Path(__file__).parent/"logs/AdsCO2_AdsN2_seed42_cgcnn/version_5"
    # model_dir = Path(__file__).parent/"logs/AdsCO2_AdsN2_AdsS_QstCO2_QstN2_seed42_cgcnn/version_1"
    # model_dir = Path(__file__).parent/"logs/logAdsCO2_logAdsN2_logAdsS_seed42_cgcnn/version_10"
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_5"  ## GMOF
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_7"  ## GCluster
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_9"  ## GMOF, with softplus output, no selectivity loss
    model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_10"  ## GMOF, with softplus output, with selectivity loss
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_1"  ## GMOF with langmuir gate
    model_name = model_dir.parent.name + "_" + model_dir.name
    result_dir = Path(os.getcwd())/f"inference/{notes}"
    result_dir.mkdir(exist_ok=True, parents=True)
    co2frac = [0, 
               0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
               ]
    press = [
        0.0001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
    
    results = inference(cif_list, co2frac, press, model_dir, saved_dir=result_dir, clean=clean)
    
    df_res = pd.DataFrame({k:v for k,v in results.items() if k.endswith("Predicted") or k in ["CO2Fraction", "Pressure[bar]"]}, index=results["CifId"])
    df_res.index.name = "MofName"
    print(df_res)
    df_res.to_csv(Path(result_dir)/f"infer_results_{model_name}.csv", float_format='%.6f')