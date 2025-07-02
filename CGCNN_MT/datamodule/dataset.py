'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-09-04 17:16:58
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

from __future__ import print_function, division

import functools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class LoadGraphData(Dataset):
    """ 
    Load CIFDATA dataset from "CIF_NAME.graphdata"
    """
    def __init__(self, data_dir, split, radius=8, dmin=0, step=0.2, 
                 prop_cols=None, use_cell_params=False, use_extra_fea=False,
                 task_id=0, **kwargs
                 ):
        
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.csv_file_name = kwargs.get("csv_file_name", f"{split}.csv")
        self.cifid_col = kwargs.get("cifid_col", "MofName")
        self.condi_cols = kwargs.get("condi_cols", ["Pressure[bar]", "CO2Fraction"])
        self.log_press = kwargs.get("log_press", True)
        self.max_num_nbr = 12

        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        
        self.data_dir = data_dir

        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split)
        self.prop_cols = [col for col in self.prop_cols if col in self.id_prop_df.columns]
        assert len(self.prop_cols) > 0, "No property columns found in the csv file"
        print(f"prop_cols of {task_id}: {self.prop_cols}")

        # self.id_prop_df.fillna(0, inplace=True)
        if self.condi_cols is not None and "Pressure[bar]" in self.condi_cols and self.log_press:
            self.id_prop_df["Pressure[bar]"] = np.log10((self.id_prop_df["Pressure[bar]"].values)+1e-5)
            print("Convert pressure to log10(x+1e-5) unit")

        # for col in ["AdsCO2", "AdsN2"]:
        #     if col in self.prop_cols:
        #         self.id_prop_df[col] = np.log10(self.id_prop_df[col].values+1e-5)
        #         print(f"Convert {col} to log10(x+1e-5) unit")

        self.g_data = {}
        for cif_id in self.id_prop_df[self.cifid_col].unique():
            g_file = data_dir / "graphs" / f"{cif_id}.graphdata"
            assert g_file.exists(), f"Graph data not found for {cif_id}"
            with open(g_file, 'rb') as f:
                data = pickle.load(f)
            self.g_data[cif_id] = data
        print(f"Load {len(self.g_data)} graph data")
        
        assert len(self.g_data) == len(self.id_prop_df[self.cifid_col].unique()), f'{len(self.g_data)} != {len(self.id_prop_df[self.cifid_col].unique())}'

        atom_prop_json = Path(__file__).parent/'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)
        self.g_data.update(new_data.g_data)

    def __len__(self):
        return len(self.id_prop_df)

    @functools.lru_cache(maxsize=1024)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,LCD,PLD,Desity(g/cm^3),VSA(m^2/cm^3),GSA(m^2/g),Vp(cm^3/g),VoidFraction,Label
        cif_id = row[self.cifid_col]

        if self.use_extra_fea:
            extra_fea = row.loc[self.condi_cols].values.astype(float)
        else:
            extra_fea = []
    
        targets = row[self.prop_cols].values.astype(float)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = self.g_data[cif_id]
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == self.max_num_nbr, \
        f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= {self.max_num_nbr} for file: {self.g_data[cif_id]}"

        targets = torch.FloatTensor(targets)

        extra_fea = torch.FloatTensor(extra_fea)

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
            "targets": targets,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        del atom_fea
        del nbr_fea_idx
        del nbr_dist
        del nbr_fea
        del extra_fea
        del targets

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["targets"]

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
        dict_batch["targets"] = torch.stack(batch_targets, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])

        del batch_atom_fea
        del batch_nbr_fea_idx
        del batch_nbr_fea
        del batch_extra_fea
        del batch_targets

        return dict_batch


class LoadGraphDataWithAtomicNumber(Dataset):

    """ 
    Load CIFDATA dataset from "CIF_NAME.graphdata"
    """
    def __init__(self, data_dir, split, radius=8, dmin=0, step=0.2, 
                 prop_cols=None, use_cell_params=False, use_extra_fea=False,
                 task_id=0, **kwargs
                 ):
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.csv_file_name = kwargs.get("csv_file_name", f"{split}.csv")
        self.cifid_col = kwargs.get("cifid_col", "MofName")
        self.condi_cols = kwargs.get("condi_cols", ["Pressure[bar]", "CO2Fraction"])
        self.log_press = kwargs.get("log_press", True)
        self.max_num_nbr = 12

        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split)
        self.prop_cols = [col for col in self.prop_cols if col in self.id_prop_df.columns]
        assert len(self.prop_cols) > 0, "No property columns found in the csv file"
        print(f"prop_cols of {task_id}: {self.prop_cols}")

        # self.id_prop_df.fillna(0, inplace=True)
        if self.condi_cols is not None and "Pressure[bar]" in self.condi_cols and self.log_press:
            self.id_prop_df["Pressure[bar]"] = np.log10((self.id_prop_df["Pressure[bar]"].values)+1e-5)
            print("Convert pressure to log10(x+1e-5) unit")
        # for col in ["AdsCO2", "AdsN2"]:
        #     if col in self.prop_cols:
        #         self.id_prop_df[col] = np.log10(self.id_prop_df[col].values+1e-5)
        #         print(f"Convert {col} to log10(x+1e-5) unit")
        self.g_data = {}
        for cif_id in self.id_prop_df[self.cifid_col].unique():
            g_file = data_dir / "graphs" / f"{cif_id}.graphdata"
            assert g_file.exists(), f"Graph data not found for {cif_id}"
            with open(g_file, 'rb') as f:
                data = pickle.load(f)
            self.g_data[cif_id] = data
        print(f"Load {len(self.g_data)} graph data")
        
        assert len(self.g_data) == len(self.id_prop_df[self.cifid_col].unique()), f'{len(self.g_data)} != {len(self.id_prop_df[self.cifid_col].unique())}'

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)
        self.g_data.update(new_data.g_data)

    def __len__(self):
        return len(self.id_prop_df)
    
    @functools.lru_cache(maxsize=1024)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,LCD,PLD,Desity(g/cm^3),VSA(m^2/cm^3),GSA(m^2/g),Vp(cm^3/g),VoidFraction,Label
        cif_id = row[self.cifid_col]

        if self.use_extra_fea:
            extra_fea = row.loc[self.condi_cols].values.astype(float)
        else:
            extra_fea = []
        
        targets = row[self.prop_cols].values.astype(float)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, uni_idx, uni_count, cell_params = self.g_data[cif_id]

        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == self.max_num_nbr, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= {self.max_num_nbr} for file: {self.g_data[cif_id]}"

        targets = torch.FloatTensor(targets)

        extra_fea = torch.FloatTensor(extra_fea)

        atom_fea = torch.LongTensor(atom_num) ## use atomic number as feature

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
            "uni_idx": uni_idx,
            "uni_count": uni_count,
            "extra_fea": extra_fea,
            "targets": targets,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        del atom_fea
        del nbr_fea
        del nbr_fea_idx
        del extra_fea
        del targets

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["targets"]

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
        dict_batch["targets"] = torch.stack(batch_targets, dim=0)
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])

        del batch_atom_fea
        del batch_nbr_fea_idx
        del batch_nbr_fea
        del batch_extra_fea
        del batch_targets

        return dict_batch
    
    
def sample_data(id_prop_file, split):
    
    """
    Sample data from dataset
    """
    
    assert os.path.exists(id_prop_file), f'{str(id_prop_file)} not exists'
    id_prop_df = pd.read_csv(id_prop_file)
    if split not in ["train", "val", "test"] or split in str(id_prop_file):
        return id_prop_df
    id_prop_df = id_prop_df[id_prop_df["Partition"] == split]
    return id_prop_df


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

