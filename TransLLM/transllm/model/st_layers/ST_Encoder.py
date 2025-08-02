from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph
from transformers.configuration_utils import PretrainedConfig
from transllm.model.st_layers.args import parse_args
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from fastdtw import fastdtw

tpd = 288
class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self):
        raise NotImplementedError


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 3, 2, 1)
        return y
    
class STBlock(nn.Module):
    """Spatio-temporal encoding base module, consisting of a single-layer GAT and a single-layer TCN"""
    def __init__(self, in_channels, hidden_dim, num_heads, dropout):
        super().__init__()
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=hidden_dim, dropout=dropout)

        self.gat = GATConv(
            in_channels=hidden_dim[-1],
            out_channels=hidden_dim[-1],
            heads=num_heads,
            dropout=dropout
        )
        self.node_embedding1 = nn.Linear(18, hidden_dim[-1])
        self.node_embedding2 = nn.Linear(3, hidden_dim[-1])
        self.node_embedding4 = nn.Linear(1, hidden_dim[-1])
        self.temporal2 = TemporalConvNet(num_inputs=hidden_dim[-1]*num_heads,
                                   num_channels=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim[-1])
    def forward(self, x, edge_index,nodes_feature, B, T, N):
        """
        x: [B*T*N, in_channels]
        return: [B*T*N, hidden_dim]
        """
        h = self.temporal1(x.permute(0,2,1,3))
        if nodes_feature is not None:
            Type = nodes_feature.shape[1]
            if Type == 18:
                node_embeddings = self.node_embedding1(nodes_feature)
            elif Type == 3:
                node_embeddings = self.node_embedding2(nodes_feature)
            else:
                node_embeddings = self.node_embedding4(nodes_feature)
            node_embeddings = node_embeddings[None, None, :, :]  # shape: [1, 1, 673, 64]
            h = h + node_embeddings
        
        gat_outputs=[]
        for t_step in range(T):
            #  # Get features at the current time step [B, N, hidden]
            h_t = h[:, t_step, :, :]
            
            # Flatten batch and node dimensions
            h_flat = h_t.reshape(B*N, -1)  # [B*N, hidden]
            
            h_gat = self.gat(h_flat, edge_index)  # [B*N, out_channels]
            
            #  [B, N, out_channels]
            h_gat = h_gat.view(B, N, -1)
            gat_outputs.append(h_gat)
        
        gat_outputs = torch.stack(gat_outputs, dim=1)
        outs = self.temporal2(gat_outputs.permute(0,2,1,3))
        h = self.norm(F.relu(outs.reshape(B,T,N, -1)))
        return h     # [B*T*N, H]

class ST_Enc(BaseModel):
    def __init__(self, 
                 device,
                 horizon=12,
                 hidden_dim=64, 
                 num_heads=2,
                 num_blocks=2, 
                 max_neighbors=20,
                 dropout=0.2,
                 time_stride=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = PretrainedConfig()

        self.horizon = horizon
        
        self.blocks1 = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = self.input_dim if i == 0 else hidden_dim
            self.blocks1.append(STBlock(
                in_channels=in_channels,
                hidden_dim=[64, 32, 64],
                num_heads=num_heads,
                dropout=dropout
            ))
        self.blocks2 = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = self.input_dim if i == 0 else hidden_dim
            self.blocks2.append(STBlock(
                in_channels=in_channels,
                hidden_dim=[64, 32, 64],
                num_heads=num_heads,
                dropout=dropout
            ))
        self.device = device
        self.pred1 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )
        self.pred2 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )
        self.pred3 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, 1)
        )
        self.pred4 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, 1)
        )
        self.pred5 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )
        self.pred6 = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim*num_heads, self.horizon * 32), 
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )
    def forward(self, x, sp_matrix,se_matrix,type,nodes_feature=None, label=None):
        """
        x: [B, T, N, C]
        return: [B, horizon, N, 1]
        """
        B, T, N, C = x.shape
        h1 = h2 = x
        for block in self.blocks1:
            h1 = block(h1, sp_matrix,nodes_feature, B, T, N)
        for block in self.blocks2:
            h2 = block(h2, se_matrix,nodes_feature, B, T, N)

        x = torch.concat([h1,h2],dim=-1)
        x_emb = x
        x = x.permute(0,2,1,3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        if type ==1:
            x = self.pred1(x)
        elif type ==2:
            x = self.pred2(x)
        elif type ==3:
            x = self.pred3(x)
        elif type==4:
            x = self.pred4(x)
        elif type==5:
            x = self.pred5(x)
        elif type==6:
            x = self.pred6(x)
        x = x.unsqueeze(-1).transpose(1, 2)
        return x, x_emb
