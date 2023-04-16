import math
import os
import sys
import random
import logging
import torch
import torch.utils.data
import torchaudio
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

sys.path.append("../")
from config import MAX_WAV_VALUE
from utils.utils import pad_or_trim
from datasets.dataset import BaseDataset


mel_basis = {}
hann_window = {}

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class SVCGANDataset(BaseDataset):
    
    def __init__(
        self, dataset, dataset_type, args,
        n_fft, num_mels,hop_size, win_size, sampling_rate,  
        fmin, fmax, fmax_loss=None, mel_crop_length=800, 
        audio_crop_length=256000,
    ):
        
        BaseDataset.__init__(self,dataset, dataset_type, args)
        
        
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.mel_crop_length = mel_crop_length
        self.audio_crop_length = audio_crop_length

    
    def __getitem__(self, idx):
      
          
        audio = self.waveform[idx]
        
        audio, _ = pad_or_trim(audio,length=self.audio_crop_length)

        audio_mask = self.waveform_mask[idx]
        
        audio_mask = audio_mask[:,:self.audio_crop_length]
        audio = audio / MAX_WAV_VALUE
        
        audio = normalize(audio.squeeze(0).numpy()) * 0.95

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        mel = mel_spectrogram(audio[audio_mask==1].unsqueeze(0), self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size,
                              self.win_size, self.fmin, self.fmax,
                                  center=False)

        mel, mel_mask = pad_or_trim(mel, self.mel_crop_length,axis=-1)
        
        mel_loss = mel_spectrogram(audio[audio_mask==1].unsqueeze(0), self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, 
                                   self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
       
        mel_loss, mel_loss_mask = pad_or_trim(mel_loss, self.mel_crop_length,axis=-1)
        
        sample = (mel.squeeze(), 
                  audio.squeeze(0),  
                  mel_loss.squeeze(),
                  mel_mask.squeeze(),
                  torch.FloatTensor(audio_mask.squeeze(0)), 
                  torch.FloatTensor(self.lf0[idx]), 
                  torch.FloatTensor(self.whisper[idx]), 
                  torch.FloatTensor(self.hubert[idx]))
        return sample