'''
Author: zhangshd
Date: 2024-08-16 11:08:17
LastEditors: zhangshd
LastEditTime: 2024-11-11 18:43:20
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodule.dataset import LoadGraphData, LoadGraphDataWithAtomicNumber
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


class DInterface(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=64, num_workers=4, dataset_cls=LoadGraphData,
                 **kwargs):
        
        super().__init__()
        self.root_dir = Path(data_dir)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tasks = kwargs.get('tasks', ['AdsCO2', 'AdsN2', 'QstCO2', 'QstN2'])
        self.task_types = kwargs.get('task_types', ['regression_log', 'regression_log', 'regression', 'regression'])
        
        self.kwargs = kwargs
        self.dataset_cls = dataset_cls
        if isinstance(dataset_cls, str):
            self.dataset_cls = eval(dataset_cls)
        self.collate_fn = self.dataset_cls.collate

    def setup(self, stage=None):
        # This method is called on every GPU

        # Initialize CIFData dataset
        if stage == 'fit' or stage is None:
            if not hasattr(self, 'trainset'):
                self.trainset = self.dataset_cls(data_dir=self.root_dir, split='train', 
                                            prop_cols=self.tasks, **self.kwargs)
                    
                print("Number of total training data:", len(self.trainset))
                self.task_weights = (self.trainset.id_prop_df[self.trainset.prop_cols].notna().sum() / len(self.trainset))
                self.task_weights = list(self.task_weights / self.task_weights.sum())
                
                print("Task weights:", self.task_weights)
            self.train_normalizer()
            if not hasattr(self, 'valset'):
                self.valset = self.dataset_cls(data_dir=self.root_dir, split='val',
                                        prop_cols=self.tasks, **self.kwargs)
                print("Number of total validation data:", len(self.valset))

        if (stage == 'test' or stage is None) and not hasattr(self, 'testset'):
            self.testset = self.dataset_cls(data_dir=self.root_dir, split='test',
                                                prop_cols=self.tasks,  **self.kwargs)
            print("Number of total test data:", len(self.testset))


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)
                          

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)
    
    def train_normalizer(self):
        self.normalizers = []
        for i, task_tp in enumerate(self.task_types):
            if 'classification' in task_tp:
                normalizer = Normalizer(torch.Tensor([-1, 0., 1]))
                self.normalizers.append(normalizer)
            else:
                train_targets = torch.Tensor(self.trainset.id_prop_df.loc[:, self.trainset.prop_cols[i]].values)
                if "log" in task_tp:
                    normalizer = Normalizer(train_targets, log_labels=True)
                else:
                    normalizer = Normalizer(train_targets)
                self.normalizers.append(normalizer)
        return self.normalizers

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, log_labels=False, remove_value=None):
        """tensor is taken as a sample to calculate the mean and std"""
        super(Normalizer, self).__init__()
        self.log_labels = log_labels
        tensor = tensor[torch.isnan(tensor) == False]  # remove NaN values for normalization
        if remove_value is not None:
            tensor = tensor[tensor != remove_value]  # remove 0 values for normalization
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5) # avoid log10(0)
            print("Log10(x+1e-5) transform applied to labels.")
        
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        self.mean_ = float(self.mean.cpu().numpy())
        self.std_ = float(self.std.cpu().numpy())
        self.device = tensor.device

    def norm(self, tensor):
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        denormed_tensor = normed_tensor * self.std + self.mean
        if hasattr(self, 'log_labels') and self.log_labels:
            denormed_tensor = torch.clamp(denormed_tensor, -20, 20)  # avoid numerical errors
            return torch.pow(10, denormed_tensor) - 1e-5
        else:
            return denormed_tensor

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.mean_ = float(self.mean.cpu().numpy())
        self.std_ = float(self.std.cpu().numpy())
        
    def to(self, device):
        """Moves both mean and std to the specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device

        return self  
