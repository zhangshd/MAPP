# MOFTransformer version 2.0.0
import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPredictionHeadTransform,
)


class Pooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GGMHead(nn.Module):
    """
    head for Graph Grid Matching
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MPPHead(nn.Module):
    """
    head for Masked Patch Prediction (regression version)
    """

    def __init__(self, hid_dim):
        super().__init__()

        bert_config = BertConfig(
            hidden_size=hid_dim,
        )
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(hid_dim, 101 + 2)  # bins

    def forward(self, x):  # [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)  # [B, max_len, bins]
        return x


class MTPHead(nn.Module):
    """
    head for MOF Topology Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1397)  # len(assets/topology.json)

    def forward(self, x):
        x = self.fc(x)
        return x


class VFPHead(nn.Module):
    """
    head for Void Fraction Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x


class RegressionHead(nn.Module):
    """
    head for Regression
    """

    def __init__(self, hid_dim, Softplus=False):
        super().__init__()
        self.fc_out = nn.Linear(hid_dim, 1)
        self.softplus = Softplus

    def forward(self, x):
        x = self.fc_out(x)
        if self.softplus:
            x = torch.nn.functional.softplus(x) ## make sure the output is non-negative
        return x


class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()

        if n_classes == 2:
            self.fc_out = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc_out = nn.Linear(hid_dim, n_classes)
            self.binary = False

    def forward(self, x):
        x = self.fc_out(x)

        return x, self.binary


class MOCHead(nn.Module):
    """
    head for Metal Organic Classification
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        """
        :param x: graph_feats [B, graph_len, hid_dim]
        :return: [B, graph_len]
        """
        x = self.fc(x)  # [B, graph_len, 1]
        x = x.squeeze(dim=-1)  # [B, graph_len]
        return x

## adapted from Uni-MOF
class ExtraEmbedding3D(nn.Module):
    def __init__(self, hidden_dim, bins=32, orig_extra_dim=2, min_max_key=None):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.project1 = nn.Linear(1, hidden_dim//2)
        self.norm1 = nn.BatchNorm1d(orig_extra_dim)
        self.project2 = NonLinearHead(hidden_dim, hidden_dim)
        self.bins = bins
        self.pressure_embed = nn.Embedding(bins, hidden_dim//2)
        self.co2frac_embed = nn.Embedding(bins, hidden_dim//2)
        self.min_max_key = min_max_key
        
    def forward(self, extra_fea):
        
        extra_fea = extra_fea.type_as(self.project1.weight)

        pressure = extra_fea[:, 0]  ## pressure is the first feature
        pressure = torch.clamp(pressure, self.min_max_key['pressure'][0], self.min_max_key['pressure'][1])
        pressure = (pressure - self.min_max_key['pressure'][0]) / (self.min_max_key['pressure'][1] - self.min_max_key['pressure'][0])
        pressure_bin = torch.floor(pressure * self.bins).to(torch.long)
        pressure_embed = self.pressure_embed(pressure_bin)  # [B, hidden_dim//2]

        co2frac = extra_fea[:, 1]  ## co2frac is the second feature
        co2frac = torch.clamp(co2frac, self.min_max_key['co2frac'][0], self.min_max_key['co2frac'][1])
        co2frac = (co2frac - self.min_max_key['co2frac'][0]) / (self.min_max_key['co2frac'][1] - self.min_max_key['co2frac'][0])
        co2frac_bin = torch.floor(co2frac * self.bins).to(torch.long)
        co2frac_embed = self.co2frac_embed(co2frac_bin) # [B, hidden_dim//2]

        env_project = self.project1(extra_fea.unsqueeze(2))  # [B, orig_extra_dim, hidden_dim//2]
        env_project = self.norm1(env_project)
        embed_repr = torch.cat([pressure_embed.unsqueeze(1), co2frac_embed.unsqueeze(1)] + [
            torch.zeros_like(pressure_embed.unsqueeze(1)) for i in range(extra_fea.size(1)-2)
            ], dim=1) # [B, orig_extra_dim, hidden_dim//2]
        env_repr = torch.cat([env_project, embed_repr], dim=-1) # [B, orig_extra_dim, hidden_dim]
        env_repr = self.project2(env_repr) # [B, orig_extra_dim, hidden_dim]

        return env_repr
    

## adapted from Uni-MOF
class ExtraEmbedding(nn.Module):
    def __init__(self, hidden_dim, bins=32, orig_extra_dim=2, min_max_key=None):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.project1 = NonLinearHead(orig_extra_dim, hidden_dim, hidden=hidden_dim, norm="batch")
        self.project2 = NonLinearHead(hidden_dim*2, hidden_dim)
        self.bins = bins
        self.pressure_embed = nn.Embedding(bins, hidden_dim//2)
        self.co2frac_embed = nn.Embedding(bins, hidden_dim//2)
        self.min_max_key = min_max_key
        
    def forward(self, extra_fea):
        
        extra_fea = extra_fea.type_as(self.project1.linear1.weight)

        pressure = extra_fea[:, 0]  ## pressure is the first feature
        pressure = torch.clamp(pressure, self.min_max_key['pressure'][0], self.min_max_key['pressure'][1])
        pressure = (pressure - self.min_max_key['pressure'][0]) / (self.min_max_key['pressure'][1] - self.min_max_key['pressure'][0])
        pressure_bin = torch.floor(pressure * self.bins).to(torch.long)
        pressure_embed = self.pressure_embed(pressure_bin)  # shape of pressure_embed is [batch_size, env_dim//2]

        co2frac = extra_fea[:, 1]  ## co2frac is the second feature
        co2frac = torch.clamp(co2frac, self.min_max_key['co2frac'][0], self.min_max_key['co2frac'][1])
        co2frac = (co2frac - self.min_max_key['co2frac'][0]) / (self.min_max_key['co2frac'][1] - self.min_max_key['co2frac'][0])
        co2frac_bin = torch.floor(co2frac * self.bins).to(torch.long)
        co2frac_embed = self.co2frac_embed(co2frac_bin)

        env_project = self.project1(extra_fea)  # shape of env_project is [batch_size, env_dim//2]
        env_repr = torch.cat([env_project, pressure_embed, co2frac_embed], dim=-1)
        env_repr = self.project2(env_repr)

        return env_repr

class NonLinearHead(nn.Module):
    """Head for simple tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden=None,
        norm="none",
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = nn.Softplus()
        if norm == "batch":
            self.norm = nn.BatchNorm1d(hidden)
        elif norm == "layer":
            self.norm = nn.LayerNorm(hidden)
        else:
            self.norm = nn.Identity()

    def forward(self, x):

        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class LangmuirGatedRegressionHead(nn.Module):
    """
    Langmuir-gated regression head for adsorption isotherm prediction.
    
    Ensures thermodynamic consistency:
    - q(P=0) = 0 (vacuum boundary condition)
    - q(P→∞) → q_sat (saturation limit)
    
    Architecture: q_hat = Gate(P_partial) * N(features)
    where Gate(P) = P / (1 + b*P) is the Langmuir gate factor.
    
    Args:
        hid_dim: Hidden dimension from transformer
        learnable_b: Whether the saturation parameter b is learnable
        b_init: Initial value for b parameter (in 1/bar units)
        use_softplus_output: Apply softplus to ensure non-negative output
        component: 'CO2' or 'N2' - determines which partial pressure to use
        arcsinh_pressure_idx: Index of arcsinh(P) in extra_fea
        co2_fraction_idx: Index of CO2 fraction in extra_fea
    """
    
    def __init__(self, hid_dim, 
                 learnable_b=True, 
                 b_init=1.0,
                 use_softplus_output=True,
                 component='CO2',
                 arcsinh_pressure_idx=0,
                 co2_fraction_idx=2,
                 power=1.0,
                 learnable_power=False,
                 power_min=1.0,
                 power_max=5.0):
        super().__init__()
        self.fc_out = nn.Linear(hid_dim, 1)
        self.use_softplus_output = use_softplus_output
        self.component = component.upper()
        self.arcsinh_pressure_idx = arcsinh_pressure_idx
        self.co2_fraction_idx = co2_fraction_idx
        self.learnable_b = learnable_b
        self.learnable_power = learnable_power
        self.power_min = power_min
        self.power_max = power_max
        
        # b parameter: saturation parameter
        if learnable_b:
            # Using raw parameter + softplus to ensure b > 0
            self.b_raw = nn.Parameter(torch.tensor(float(b_init)))
        else:
            self.register_buffer('b', torch.tensor(float(b_init)))
        
        # power parameter: gate steepness (P^n / (1 + b*P^n))
        if learnable_power:
            # Map power to [power_min, power_max] using sigmoid
            # power = power_min + (power_max - power_min) * sigmoid(power_raw)
            # Initialize power_raw so that sigmoid(power_raw) gives (power - power_min) / (power_max - power_min)
            init_sigmoid = (power - power_min) / (power_max - power_min + 1e-8)
            init_sigmoid = max(0.01, min(0.99, init_sigmoid))  # Clip to avoid inf
            power_raw_init = -torch.log(torch.tensor(1.0 / init_sigmoid - 1.0))
            self.power_raw = nn.Parameter(power_raw_init)
        else:
            self._power = power  # Store as regular attribute
    
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
            cls_feats: [B, hid_dim] - pooled transformer features
            extra_fea: [B, extra_dim] - contains arcsinh(P), log(P), CO2fraction
        Returns:
            output: [B] - Langmuir-gated adsorption prediction
        """
        # Neural network output (represents q_sat-like value)
        nn_out = self.fc_out(cls_feats).squeeze(-1)  # [B]
        
        if self.use_softplus_output:
            nn_out = torch.nn.functional.softplus(nn_out)  # Ensure non-negative
        
        # Compute partial pressure from extra features
        P_partial = self.compute_partial_pressure(extra_fea)  # [B]
        
        # Apply Langmuir gate
        gate = self.langmuir_gate(P_partial)  # [B]
        output = gate * nn_out  # [B]
        
        return output