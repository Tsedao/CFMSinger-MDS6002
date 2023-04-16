import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MelDecoder(nn.Module):
    
    def __init__(
            self, 
            lf0_dims=5,
            ppg_dims=512,
            phoneme_dims=512,
            output_dims=80, 
            dropout=0.1, 
            nhead=8,
            nlayers=1
    ):
        super().__init__()
        input_dims = lf0_dims + ppg_dims + phoneme_dims
        encoder_layers = TransformerEncoderLayer(
            input_dims, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.mlp_mel_1 = nn.Linear(input_dims, output_dims)
        self.mlp_mel_2 = nn.Linear(output_dims, output_dims)
        self.mlp_condition = nn.Linear(input_dims, output_dims)
        
        
    def forward(
        self, 
        lf0_embed, 
        ppg_embed, 
        phoneme_embed
    ):
        """x: [N,T,C] channel last"""
        embed = torch.cat([lf0_embed, ppg_embed, phoneme_embed], dim=-1)
        
        tran_embed = self.transformer_encoder(embed)
        mel_rec = self.mlp_mel_1(tran_embed)
        
        mel_rec_cond = self.mlp_mel_2(F.leaky_relu(mel_rec))
        
        cond = self.mlp_condition(embed)
        
        cond = cond + mel_rec_cond
        
        ## permute the tensor to the mel-shape 
        cond = torch.permute(cond, dims=(0,2,1))
        mel_rec = torch.permute(mel_rec, dims=(0,2,1))
        return cond, mel_rec        