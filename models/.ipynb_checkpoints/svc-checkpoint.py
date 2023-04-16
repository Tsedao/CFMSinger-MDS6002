import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SVC(nn.Module):
    
    def __init__(
        self, 
        lf0_encoder,
        ppg_encoder,
        pho_encoder,
        decoder,
        vocoder,
    ):
        super().__init__()
        self.lf0_encoder = lf0_encoder
        self.ppg_encoder = ppg_encoder
        self.pho_encoder = pho_encoder
        self.decoder = decoder
        self.vocoder = vocoder
        
    def forward(self, lf0, ppg, pho):
        """
        Args: 
            lf0 : [N,1,T] channel first
            ppg : [N,T,WHISPER_DIM] channel last
            pho : [N,T,HUBERT_DIM] channel last
        """
        
        lf0_embed = self.lf0_encoder(lf0)
        ppg_embed = self.ppg_encoder(ppg)
        pho_embed = self.pho_encoder(pho)
        
        cond_signal, mel_rec = self.decoder(
                            torch.permute(lf0_embed,dims=(0,2,1)),
                                ppg_embed,
                                pho_embed
                                    )    # [N,T,F(C1+C2+C3)], [N,MEL_DIM,T]
        
        waveform = self.vocoder(cond_signal)
        
        return waveform, mel_rec 