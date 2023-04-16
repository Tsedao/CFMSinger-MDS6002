import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
import numpy as np


import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

matplotlib.use("Agg")
import matplotlib.pylab as plt



def plot_spectrogram(spectrogram,title=None):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title(title or "Spectrogram (db)")
    ax.set_ylabel("freq_bin")
    ax.set_xlabel("frame")
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)