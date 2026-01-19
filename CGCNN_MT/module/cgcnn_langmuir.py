"""
CGCNN with Langmuir-gated regression heads for adsorption prediction.
Inherits from CrystalGraphConvNet and replaces adsorption heads with Langmuir-gated versions.

Author: zhangshd
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from module.cgcnn import CrystalGraphConvNet
from module.layers import OutputLayer
from module.langmuir_head import LangmuirGatedRegressionHead


class CrystalGraphConvNetLangmuir(CrystalGraphConvNet):
    """
    CGCNN with Langmuir-gated regression heads for adsorption tasks.
    
    For tasks containing 'LOADING' or 'ADS', uses LangmuirGatedRegressionHead.
    For other tasks (Qst, classification), uses standard OutputLayer.
    
    This architecture ensures thermodynamic consistency:
    - q(P=0) = 0 (vacuum boundary condition)
    - q(P→∞) → q_sat (saturation limit)
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, orig_extra_fea_len, 
                 extra_fea_len=16, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, 
                 **kwargs):
        """
        Initialize CrystalGraphConvNetLangmuir.
        
        Additional kwargs for Langmuir gating:
            langmuir_learnable_b: Whether b is learnable (default: True)
            langmuir_b_init: Initial value for b (default: 1.0)
            output_activation: Output activation ('none', 'softplus', 'relu', 'leaky_relu')
            langmuir_power: Gate power (default: 1.0)
            langmuir_learnable_power: Whether power is learnable (default: True)
            langmuir_power_min: Min power value (default: 1.0)
            langmuir_power_max: Max power value (default: 5.0)
            langmuir_output_transform: Output transform type (default: 'symlog')
            langmuir_symlog_threshold: Symlog threshold (default: 1e-4)
            arcsinh_pressure_idx: Index of arcsinh(P) in extra_fea (default: 0)
            co2_fraction_idx: Index of CO2 fraction in extra_fea (default: 2)
            tasks: List of task names to detect adsorption tasks
        """
        # Initialize base CGCNN (this creates fc_outs with OutputLayer)
        super().__init__(
            orig_atom_fea_len, nbr_fea_len, orig_extra_fea_len,
            extra_fea_len, atom_fea_len, n_conv, h_fea_len, n_h,
            **kwargs
        )
        
        # Extract Langmuir configuration
        self.langmuir_learnable_b = kwargs.get('langmuir_learnable_b', True)
        self.langmuir_b_init = kwargs.get('langmuir_b_init', 1.0)
        self.output_activation = kwargs.get('output_activation', 'softplus')
        self.langmuir_power = kwargs.get('langmuir_power', 1.0)
        self.langmuir_learnable_power = kwargs.get('langmuir_learnable_power', True)
        self.langmuir_power_min = kwargs.get('langmuir_power_min', 1.0)
        self.langmuir_power_max = kwargs.get('langmuir_power_max', 5.0)
        self.langmuir_output_transform = kwargs.get('langmuir_output_transform', 'symlog')
        self.langmuir_symlog_threshold = kwargs.get('langmuir_symlog_threshold', 1e-4)
        self.arcsinh_pressure_idx = kwargs.get('arcsinh_pressure_idx', 0)
        self.co2_fraction_idx = kwargs.get('co2_fraction_idx', 2)
        
        # Get task names
        self.tasks = kwargs.get('tasks', [])
        
        # Replace fc_outs for adsorption tasks with Langmuir heads
        self._replace_adsorption_heads(h_fea_len)
    
    def _replace_adsorption_heads(self, h_fea_len):
        """Replace OutputLayer with LangmuirGatedRegressionHead for adsorption tasks."""
        new_fc_outs = nn.ModuleList()
        self.langmuir_task_indices = []
        
        for i, (task_tp, task_name) in enumerate(zip(self.task_types, self.tasks)):
            task_upper = task_name.upper()
            
            # Check if this is an adsorption task
            is_adsorption = ('LOADING' in task_upper or 'ADS' in task_upper) and 'regression' in task_tp
            
            if is_adsorption:
                # Determine component (CO2 or N2)
                component = 'CO2' if 'CO2' in task_upper else 'N2'
                
                # Create Langmuir-gated head
                head = LangmuirGatedRegressionHead(
                    hid_dim=h_fea_len,
                    learnable_b=self.langmuir_learnable_b,
                    b_init=self.langmuir_b_init,
                    activation=self.output_activation,
                    component=component,
                    arcsinh_pressure_idx=self.arcsinh_pressure_idx,
                    co2_fraction_idx=self.co2_fraction_idx,
                    power=self.langmuir_power,
                    learnable_power=self.langmuir_learnable_power,
                    power_min=self.langmuir_power_min,
                    power_max=self.langmuir_power_max,
                    output_transform=self.langmuir_output_transform,
                    symlog_threshold=self.langmuir_symlog_threshold
                )
                new_fc_outs.append(head)
                self.langmuir_task_indices.append(i)
                print(f"Task {i} ({task_name}): Using LangmuirGatedRegressionHead (component={component})")
            else:
                # Keep standard OutputLayer
                new_fc_outs.append(self.fc_outs[i])
                print(f"Task {i} ({task_name}): Using standard OutputLayer")
        
        self.fc_outs = new_fc_outs
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea, **kwargs):
        """
        Forward pass with Langmuir gating for adsorption tasks.
        
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
            Extra features including pressure and composition

        Returns
        -------
        outs: dict
            Predictions for each task.
        """
        # Atom embedding
        atom_fea = self.embedding_atom(atom_fea)
        
        # Extra feature embedding
        if hasattr(self, 'embedding_extra'):
            extra_fea_embedded = self.embedding_extra(extra_fea)
            extra_fea_embedded = self.embedding_extra_norm(extra_fea_embedded)
            extra_fea_embedded = self.embedding_extra_softplus(extra_fea_embedded)
        
        # Graph convolutions
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pooling
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        # Concatenate with extra features
        if hasattr(self, 'embedding_extra'):
            crys_fea = torch.cat([crys_fea, extra_fea_embedded], dim=1)
        
        # Fully connected layers
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        self.dropout(crys_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = fc(crys_fea)
                crys_fea = softplus(crys_fea)

        # Add noise if configured
        if self.noise_var is not None:
            with torch.random.fork_rng():
                torch.random.manual_seed(torch.seed() + 1)
                noise = torch.randn_like(crys_fea) * self.noise_var
            crys_fea = crys_fea + noise

        # Task-specific outputs
        outs = {}
        for i in range(self.n_tasks):
            if i in self.langmuir_task_indices:
                # Langmuir head needs extra_fea for pressure info
                output = self.fc_outs[i](crys_fea, extra_fea)
            else:
                # Standard head
                output = self.fc_outs[i](crys_fea)
            
            outs[i] = {
                'output': output,
                "last_layer_fea": crys_fea
            }
        
        return outs
