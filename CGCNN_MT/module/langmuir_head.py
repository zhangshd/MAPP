"""
LangmuirGatedRegressionHead for CGCNN models.
Adapted from MOFTransformer/module/heads.py

Author: zhangshd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x, threshold=1e-4):
    """Symmetric log transform: sign(x) * log10(1 + |x|/threshold)"""
    return torch.sign(x) * torch.log10(1 + torch.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """Inverse of symlog: sign(y) * threshold * (10^|y| - 1)"""
    return torch.sign(y) * threshold * (torch.pow(10, torch.abs(y)) - 1)


class LangmuirGatedRegressionHead(nn.Module):
    """
    Langmuir-gated regression head for adsorption isotherm prediction.
    
    Ensures thermodynamic consistency:
    - q(P=0) = 0 (vacuum boundary condition)
    - q(P→∞) → q_sat (saturation limit)
    
    Architecture: q_hat = Gate(P_partial) * NN(features)
    where Gate(P) = P^n / (1 + b*P^n) is the Langmuir gate factor.
    
    Args:
        hid_dim: Hidden dimension from backbone
        learnable_b: Whether the saturation parameter b is learnable
        b_init: Initial value for b parameter (in 1/bar units)
        activation: Output activation ('none', 'softplus', 'relu', 'leaky_relu')
        component: 'CO2' or 'N2' - determines which partial pressure to use
        arcsinh_pressure_idx: Index of arcsinh(P) in extra_fea
        co2_fraction_idx: Index of CO2 fraction in extra_fea
        power: Initial power value for gate (P^n / (1 + b*P^n))
        learnable_power: Whether power is learnable
        power_min: Minimum power value when learnable
        power_max: Maximum power value when learnable
        output_transform: 'none', 'symlog', or 'arcsinh'
        symlog_threshold: Threshold for symlog transform
        leaky_relu_slope: Negative slope for leaky_relu (default: 0.01)
        relu_bias_init: Initial bias value for ReLU to prevent dead neurons (default: 1.0)
    """
    
    def __init__(self, hid_dim, 
                 learnable_b=True, 
                 b_init=1.0,
                 activation='softplus',  # Default to softplus for backward compatibility
                 component='CO2',
                 arcsinh_pressure_idx=0,
                 co2_fraction_idx=2,
                 power=1.0,
                 learnable_power=False,
                 power_min=1.0,
                 power_max=5.0,
                 output_transform='none',
                 symlog_threshold=1e-4,
                 leaky_relu_slope=0.01,
                 relu_bias_init=1.0):
        super().__init__()
        self.fc_out = nn.Linear(hid_dim, 1)
        self.activation = activation.lower() if activation else 'none'
        self.leaky_relu_slope = leaky_relu_slope
        self.component = component.upper()
        self.arcsinh_pressure_idx = arcsinh_pressure_idx
        self.co2_fraction_idx = co2_fraction_idx
        self.learnable_b = learnable_b
        self.learnable_power = learnable_power
        self.power_min = power_min
        self.power_max = power_max
        
        # Initialize bias for ReLU to prevent dead neurons
        if self.activation == 'relu' and relu_bias_init > 0:
            nn.init.constant_(self.fc_out.bias, relu_bias_init)
        
        # b parameter: saturation parameter
        if learnable_b:
            # Using raw parameter + softplus to ensure b > 0
            self.b_raw = nn.Parameter(torch.tensor(float(b_init)))
        else:
            self.register_buffer('b', torch.tensor(float(b_init)))
        
        # power parameter: gate steepness (P^n / (1 + b*P^n))
        if learnable_power:
            # Map power to [power_min, power_max] using sigmoid
            init_sigmoid = (power - power_min) / (power_max - power_min + 1e-8)
            init_sigmoid = max(0.01, min(0.99, init_sigmoid))
            power_raw_init = -torch.log(torch.tensor(1.0 / init_sigmoid - 1.0))
            self.power_raw = nn.Parameter(power_raw_init)
        else:
            self._power = power
        
        # Output transform: 'none', 'symlog', or 'arcsinh'
        self.output_transform = output_transform.lower()
        self.symlog_threshold = symlog_threshold
    
    @property
    def b(self):
        """Get b parameter, ensuring it's positive."""
        if self.learnable_b:
            return F.softplus(self.b_raw)
        return self._buffers['b']
    
    @property
    def power(self):
        """Get power parameter, constrained to [power_min, power_max]."""
        if self.learnable_power:
            return self.power_min + (self.power_max - self.power_min) * torch.sigmoid(self.power_raw)
        return self._power
    
    def compute_partial_pressure(self, extra_fea):
        """
        Recover original pressure and compute partial pressure.
        
        Args:
            extra_fea: [B, extra_dim] tensor containing pressure and fraction info
        Returns:
            P_partial: [B] tensor of component partial pressure in bar
        """
        arcsinh_pressure = extra_fea[:, self.arcsinh_pressure_idx]
        
        # Recover total pressure: P = sinh(arcsinh(P))
        P_total = torch.sinh(arcsinh_pressure)  # [B], in bar
        
        # Get CO2 fraction
        if extra_fea.shape[1] > self.co2_fraction_idx:
            co2_fraction = extra_fea[:, self.co2_fraction_idx]
        else:
            # Fallback: assume CO2 if task is CO2, else N2
            co2_fraction = torch.ones_like(P_total) if 'CO2' in self.component else torch.zeros_like(P_total)
        
        # Compute partial pressure based on component
        if 'CO2' in self.component:
            P_partial = P_total * co2_fraction
        else:  # N2 or other
            P_partial = P_total * (1.0 - co2_fraction)
        
        return P_partial
    
    def langmuir_gate(self, P_partial):
        """
        Compute Langmuir gate factor: P^n / (1 + b*P^n).
        
        Physical properties:
        - P→0: Gate→0 (vacuum limit)
        - P→∞: Gate→1/b (saturation limit)
        - Higher power (n>1) makes gate steeper at low pressure
        
        Args:
            P_partial: Component partial pressure [B] in bar
        Returns:
            Gate factor [B]
        """
        epsilon = 1e-8  # Numerical stability
        if self.power != 1.0:
            P_powered = torch.pow(P_partial + epsilon, self.power)
        else:
            P_powered = P_partial
        return P_powered / (1.0 + self.b * P_powered + epsilon)
    
    def forward(self, cls_feats, extra_fea):
        """
        Forward pass with Langmuir gating.
        
        Args:
            cls_feats: [B, hid_dim] - pooled features
            extra_fea: [B, extra_dim] - contains arcsinh(P), log(P), CO2fraction
        Returns:
            output: [B] - Langmuir-gated adsorption prediction
        """
        # Neural network output (represents q_sat-like value)
        nn_out = self.fc_out(cls_feats).squeeze(-1)  # [B]
        
        # Apply activation to ensure non-negative nn_out
        if self.activation == 'softplus':
            nn_out = F.softplus(nn_out)
        elif self.activation == 'relu':
            nn_out = F.relu(nn_out)
        elif self.activation == 'leaky_relu':
            # Training: use leaky_relu to maintain gradient flow
            # Inference: clamp to ensure non-negative output
            nn_out = F.leaky_relu(nn_out, negative_slope=self.leaky_relu_slope)
            if not self.training:
                nn_out = torch.clamp(nn_out, min=0)
        # 'none': no activation
        
        # Compute partial pressure from extra features
        P_partial = self.compute_partial_pressure(extra_fea)  # [B]
        
        # Apply Langmuir gate
        gate = self.langmuir_gate(P_partial)  # [B]
        output = gate * nn_out  # [B] - original scale adsorption
        
        # Apply output transform to match label scale
        if self.output_transform == 'symlog':
            output = symlog(output, self.symlog_threshold)
        elif self.output_transform == 'arcsinh':
            output = torch.asinh(output)
        # 'none' or default: no transform, output is in original scale
        
        return output
