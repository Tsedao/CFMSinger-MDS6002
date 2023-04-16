import pyworld as pw
import torchaudio
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath, WORLD_SAMPLE_RATE, WORLD_FRAME_SHIFT, LF0_SEQ, NUM_CHUNKS
from utils.utils import pad_or_trim

def extract_world_features_of_dataset(
    dataset, dataset_type, frame_period=WORLD_FRAME_SHIFT
):
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)
    wave_dir = dataset2wavpath[dataset]

    # Dataset
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # Save dir
    f0_dir = os.path.join(data_dir, "LF0")
    os.makedirs(f0_dir, exist_ok=True)
    wav_dir = os.path.join(data_dir, "WAV")
    os.makedirs(wav_dir, exist_ok=True)

    # Extract
    f0_features = []
    waveforms = []
    waveforms_mask = []
    # sp_features = []
    for utt in tqdm(datasets):
        uid = utt["Uid"]
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))

        if dataset == "M4Singer":
            wave_file = os.path.join(wave_dir, utt["Path"])

        f0, sp, _, _, waveform = extract_world_features(wave_file, frame_period=frame_period)
        f0 = np.where(f0!=0.0, np.log(f0),0)
        # sp_features.append(sp)
        f0, _ = pad_or_trim(f0, length=LF0_SEQ)
        waveform, wav_mask = pad_or_trim(waveform,length=WORLD_SAMPLE_RATE*NUM_CHUNKS)
        f0_features.append(f0)
        waveforms.append(waveform)
        waveforms_mask.append(wav_mask)



    # F0 statistics
    f0_statistics_file = os.path.join(f0_dir, "{}_lf0.pkl".format(dataset_type))
    
    # padded waveform
    wav_file = os.path.join(wav_dir, "{}_wav.pkl".format(dataset_type))
    wav_mask_file = os.path.join(wav_dir, "{}_wav_mask.pkl".format(dataset_type))
    
    print(f0_features[0].shape)
    print(waveforms[0].shape)
    with open(f0_statistics_file, "wb") as f:
        pickle.dump(f0_features, f)
        
    with open(wav_file, "wb") as f:
        pickle.dump(waveforms, f)
    
    with open(wav_mask_file, "wb") as f:
        pickle.dump(waveforms_mask, f)

def extract_world_features(
    wave_file, fs=WORLD_SAMPLE_RATE, frame_period=WORLD_FRAME_SHIFT
):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sample_rate, new_freq=fs
    )
    # x: (seq)
    x = np.array(waveform[0], dtype=np.double)

    _f0, t = pw.dio(x, fs, frame_period=frame_period)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)  # extract aperiodicity

    return f0, sp, ap, fs, waveform


def world_synthesis(f0, sp, ap, fs, frame_period=WORLD_FRAME_SHIFT):
    y = pw.synthesize(
        f0, sp, ap, fs, frame_period=frame_period
    )  # synthesize an utterance using the parameters
    return y


if __name__  == "__main__":
    
    extract_world_features_of_dataset("Opencpop", "test")
    extract_world_features_of_dataset("Opencpop", "train")
    extract_world_features_of_dataset("M4Singer", "test")


