'''
Author: zhangshd
Date: 2024-08-16 11:04:32
LastEditors: zhangshd
LastEditTime: 2024-11-29 23:01:22
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from module.layers import ConvLayer, OutputLayer
from torch.nn import init

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, orig_extra_fea_len, extra_fea_len=16,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, **kwargs):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()

        self.task_types = kwargs.get('task_types', ['regression', 'classification', 'regression'])
        self.tasks = kwargs.get('tasks', [])
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.loss_aggregation = kwargs.get('loss_aggregation', "sum")
        self.output_softplus = kwargs.get('output_softplus', False)  # Global softplus switch for regression
        print("task_types: ", self.task_types)
        self.n_tasks = len(self.task_types)
        self.noise_var = kwargs.get('noise_var', None)

        self.embedding_atom = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        if orig_extra_fea_len > 0:
            self.embedding_extra = nn.Linear(orig_extra_fea_len, extra_fea_len)
            self.embedding_extra_norm = nn.BatchNorm1d(extra_fea_len)
            self.embedding_extra_softplus = nn.Softplus()
            self.conv_to_fc = nn.Linear(atom_fea_len + extra_fea_len, h_fea_len)
        else:
            self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        
        self.conv_to_fc_softplus = nn.Softplus()
        self.dropout = nn.Dropout(self.dropout_prob)
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        
        # Create output layers, with softplus for regression tasks if output_softplus is enabled
        self.fc_outs = nn.ModuleList([
            OutputLayer(h_fea_len, task_tp, use_softplus=self.output_softplus and 'regression' in task_tp) 
            for task_tp in self.task_types
        ])
        
        # Track which tasks use softplus output (for skipping norm/denorm)
        self.softplus_task_indices = [i for i, task_tp in enumerate(self.task_types) 
                                       if self.output_softplus and 'regression' in task_tp]

        # define log_vars for each task
        if self.loss_aggregation == "trainable_weight_sum":
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea, **kwargs):
        """
        Forward pass.

        Parameters
        ----------

        atom_fea: torch.Tensor
          Atom features from atom type
        nbr_fea: torch.Tensor
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor
          Mapping from the crystal idx to atom idx
        extra_fea: torch.Tensor
          extra features of each crystal

        Returns
        -------

        outs: list of torch.Tensor
          Predictions for each task.
        """
        
        atom_fea = self.embedding_atom(atom_fea)
        
        if hasattr(self, 'embedding_extra'):
            extra_fea = self.embedding_extra(extra_fea)
            extra_fea = self.embedding_extra_norm(extra_fea)
            extra_fea = self.embedding_extra_softplus(extra_fea)
            
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        if hasattr(self, 'embedding_extra'):
            crys_fea = torch.cat([crys_fea, extra_fea], dim=1)
        
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = fc(crys_fea)
                crys_fea = softplus(crys_fea)

        if self.noise_var is not None:
            with torch.random.fork_rng():
                torch.random.manual_seed(torch.seed() + 1)
                noise = torch.randn_like(crys_fea) * self.noise_var  # Adding Gaussian noise with standard deviation 0.1
            crys_fea = crys_fea + noise

        outs = {}
        for i in range(self.n_tasks):
            outs[i] = {
                'output': self.fc_outs[i](crys_fea),
                "last_layer_fea": crys_fea
            }
        
        return outs

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features.

        Parameters
        ----------

        atom_fea: torch.Tensor
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)