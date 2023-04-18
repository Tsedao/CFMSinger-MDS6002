import math
import os
import sys
import random
import time
import pickle
import logging
import torch
import torch.utils.data
from librosa.util import normalize

sys.path.append("../")
from utils.utils import pad_or_trim
from config import MAX_WAV_VALUE
from datasets.dataset import BaseDataset
from datasets.dataset_gan import mel_spectrogram


class SVCDIFFDataset(BaseDataset):
    
    def __init__(
        self, dataset, dataset_type, args,
        n_fft, num_mels,hop_size, win_size, sampling_rate,  
        fmin, fmax, mel_crop_length=800, 
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
        
        
        sample = (mel.squeeze(), 
                  audio.squeeze(0),  
                  mel_mask.squeeze(),
                  torch.FloatTensor(audio_mask.squeeze(0)), 
                  torch.FloatTensor(self.lf0[idx]), 
                  torch.FloatTensor(self.whisper[idx]), 
                  torch.FloatTensor(self.hubert[idx]))
        return sample