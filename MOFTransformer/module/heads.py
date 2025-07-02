# MOFTransformer version 2.0.0
import torch.nn as nn
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

    def __init__(self, hid_dim):
        super().__init__()
        self.fc_out = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.fc_out(x)
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