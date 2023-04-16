import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
    
class LF0Encoder(nn.Module):
    """x: shape [N,C,T] channel first"""
    def __init__(self, in_dims, out_dims, kernel_size=5):
        super().__init__()
        
        self.conv = nn.Conv1d(
                        in_dims, 
                        out_dims, 
                        kernel_size=kernel_size,
                        stride=1,
                        padding='same'
                    )
        self.batch_norm = nn.BatchNorm1d(out_dims)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        
        return x
    
    
class PPGEncoder(nn.Module):
    
    def __init__(self, output_dims=80, d_model=512,dropout=0.1,nhead=8,nlayers=2):
        super().__init__()
        
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.mlp = nn.Linear(d_model, output_dims)
        
        
    def forward(self,x):
        """x: shape [N,T,C] channel last"""
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.mlp(x)
        
        return x 
    
class PhonemeEncoder(nn.Module):
    
    def __init__(self, output_dims=80, d_model=768,dropout=0.1,nhead=8,nlayers=2):
        super().__init__()
        
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.mlp = nn.Linear(d_model, output_dims)
        
        
    def forward(self,x):
        """x: shape [N,T,C] channel last"""
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.mlp(x)
        
        return x