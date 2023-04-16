import math
import os
import sys
import random
import time
import pickle
import logging
import torch
import torch.utils.data
import torchaudio
import numpy as np



sys.path.append("../")
from config import data_path

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_type, args = None):
        self.args = args
        self.dataset_type = dataset_type
        # self.y_seq_len = eval(
        #     "self.args.{}_input_length".format(self.args.model.lower())
        # )

        self.dataset_dir = os.path.join(data_path, dataset)

        logging.info("\n" + "=" * 20 + "\n")
        logging.info("{}, {} Dataset".format(dataset, dataset_type))
        self.loading_data()
        logging.info("\n" + "=" * 20 + "\n")

    def loading_whisper(self):
        logging.info("Loading Whisper features...")
        with open(
            os.path.join(self.dataset_dir, "Whisper/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.whisper = pickle.load(f)
        logging.info(
            "Whisper: sz = {}, shape = {}".format(
                len(self.whisper), self.whisper.shape
            )
        )
        
        
    def loading_hubert(self):
        logging.info("Loading Hubert features...")
        with open(
            os.path.join(self.dataset_dir, "Hubert/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.hubert = pickle.load(f)
        logging.info(
            "Hubert: sz = {}, shape = {}".format(
                len(self.hubert), self.hubert.shape
            )
        )
    
    def loading_waveform(self):
        logging.info("Loading raw wavform...")
        with open(
            os.path.join(self.dataset_dir, "WAV/{}_wav.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.waveform = pickle.load(f)
            
        with open(
            os.path.join(self.dataset_dir, "WAV/{}_wav_mask.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.waveform_mask = pickle.load(f)
            
        logging.info(
            "waveform: sz = {}, shape = {}".format(
                len(self.waveform), np.concatenate(self.waveform, axis=0).shape
            )
        )
        
    def loading_lf0(self):
        logging.info("Loading Log F0 features...")
        with open(
            os.path.join(self.dataset_dir, "LF0/{}_lf0.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.lf0 = pickle.load(f)
        logging.info(
            "LF0: sz = {}, shape = {}".format(
                len(self.lf0), np.stack(self.lf0, axis=0).shape
            )
        )
    

    def loading_data(self):
        t = time.time()

        self.loading_waveform()
        self.loading_whisper()
        self.loading_hubert()
        self.loading_lf0()

        logging.info("Done. It took {:.2f}s".format(time.time() - t))

    def __len__(self):
        return len(self.waveform)

    def __getitem__(self, idx):
        # y_gt, mask = self.get_padding_y_gt(idx)
        sample = (self.waveform[idx].squeeze(0), self.waveform_mask[idx].squeeze(0), 
                    self.lf0[idx], self.whisper[idx], self.hubert[idx])
        return sample