import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

import numpy as np

class BaseSVC(nn.Module):
    def __init__(
        self,
        lf0_encoder,
        ppg_encoder,
        pho_encoder,
        decoder,
        vocoder,
        audio_len,
    ):
        super().__init__()
        self.lf0_encoder = lf0_encoder
        self.ppg_encoder = ppg_encoder
        self.pho_encoder = pho_encoder
        self.decoder = decoder
        self.vocoder = vocoder
        self.audio_len = audio_len
        
    def get_cond(self,lf0, ppg, pho):
        
        lf0_embed = self.lf0_encoder(lf0)
        ppg_embed = self.ppg_encoder(ppg)
        pho_embed = self.pho_encoder(pho)
        
        cond_signal, mel_rec = self.decoder(
                            torch.permute(lf0_embed,dims=(0,2,1)),
                                ppg_embed,
                                pho_embed
                                    )    # [N,T,F(C1+C2+C3)], [N,MEL_DIM,T]
        
        return cond_signal, mel_rec
    
    def inference(self, lf0, ppg, pho):
        raise NotImplementedError
        
class DIFFSVC(BaseSVC):
    
    def __init__(
        self, 
        lf0_encoder,
        ppg_encoder,
        pho_encoder,
        decoder,
        vocoder,
        noise_schedule,
        inference_noise_schedule,
        audio_len = 256000
    ):
        BaseSVC.__init__(self,lf0_encoder,ppg_encoder,
                     pho_encoder,decoder,vocoder,audio_len)
        self.noise_schedule = noise_schedule
        self.inference_noise_schedule = inference_noise_schedule
                
    def forward(self, lf0, ppg, pho, t, waveform):
        """
        Args: 
            lf0 : [N,1,T] channel first
            ppg : [N,T,WHISPER_DIM] channel last
            pho : [N,T,HUBERT_DIM] channel last
        """
    
        cond_signal, mel_rec = self.get_cond(lf0, ppg, pho)
        
        noise = self.vocoder(waveform, t, cond_signal)
        
        return noise, mel_rec
    
    def inference(self, lf0, ppg, pho, fast_sampling=False):
        
        with torch.no_grad():
            cond_signal, mel_rec = self.get_cond(lf0, ppg, pho)

            training_noise_schedule = np.array(self.noise_schedule)
            inference_noise_schedule = np.array(self.inference_noise_schedule) if fast_sampling else training_noise_schedule

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)
            
            audio = torch.randn(1, self.audio_len).to(ppg)
            noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(cond_signal)


            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                audio = c1 * (audio - c2 * self.vocoder(audio, torch.tensor([T[n]], device=audio.device), cond_signal).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)
            return audio

class GANSVC(BaseSVC):
    
    def __init__(
        self, 
        lf0_encoder,
        ppg_encoder,
        pho_encoder,
        decoder,
        vocoder,
        audio_len = 256000
    ):
        BaseSVC.__init__(self,lf0_encoder,ppg_encoder,
                     pho_encoder,decoder,vocoder,audio_len)
        
    def forward(self, lf0, ppg, pho):
        """
        Args: 
            lf0 : [N,1,T] channel first
            ppg : [N,T,WHISPER_DIM] channel last
            pho : [N,T,HUBERT_DIM] channel last
        """
        
        cond_signal, mel_rec = self.get_cond(lf0, ppg, pho)
        
        waveform = self.vocoder(cond_signal)
        
        return waveform, mel_rec
    
class CFMSVC(BaseSVC):
    
    def __init__(
        self, 
        lf0_encoder,
        ppg_encoder,
        pho_encoder,
        decoder,
        vocoder,
        audio_len = 256000
    ):
        BaseSVC.__init__(self,lf0_encoder,ppg_encoder,
                     pho_encoder,decoder,vocoder,audio_len)
                
    def forward(self, lf0, ppg, pho, t, waveform):
        """
        Args: 
            lf0 : [N,1,T] channel first
            ppg : [N,T,WHISPER_DIM] channel last
            pho : [N,T,HUBERT_DIM] channel last
        """
    
        cond_signal, mel_rec = self.get_cond(lf0, ppg, pho)
        
        waveform = waveform.unsqueeze(1)
        x = (waveform, cond_signal)
        
        noise, _ = self.vocoder(t,x)
        
        return noise, mel_rec
    
    def inference(self, lf0, ppg, pho, steps=10):
        
        with torch.no_grad():
            cond_signal, _ = self.get_cond(lf0, ppg, pho)
            bs = cond_signal.shape[0]
            noise = (0.1 * torch.randn((bs, self.audio_len))).to(cond_signal)
            audio, _ = odeint(
                       self.vocoder,
                       y0=(noise,cond_signal),
                       t=torch.linspace(0,1,steps).to(cond_signal),
                       method="dopri5",
                       atol=1e-4, 
                       rtol=1e-4,
                       adjoint_options=dict(norm="seminorm")
                      )
        return audio 