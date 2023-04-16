from transformers import AutoProcessor, HubertModel

import torchaudio
import torch
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath, WORLD_SAMPLE_RATE, HUBERT_SEQ, HUBERT_DIM, NUM_CHUNKS
from utils.utils import pad_or_trim


def hubert_encoder(audio_paths):
    
    batch = len(audio_paths)
    audios = torch.zeros((batch, WORLD_SAMPLE_RATE*NUM_CHUNKS), dtype=torch.float, device=model.device)
    
    for i, audio_path in enumerate(audio_paths):
        # (48000,)
        
        audio, sr = torchaudio.load(str(audio_path))
        audio, _ = pad_or_trim(audio, length=WORLD_SAMPLE_RATE * NUM_CHUNKS)
        audios[i] = audio

    with torch.no_grad():
        # (batch, 1500, 1024)
        hubert_states = model(audios).last_hidden_state
    
    # print(hubert_states.shape)
    features, _ = pad_or_trim(hubert_states,length=HUBERT_SEQ,axis=1) 
    return features.cpu().detach().numpy()

def extract_hubert_features(dataset, dataset_type, batch_size=25):
    print("-" * 20)
    print("Dataset: {}, {}".format(dataset, dataset_type))

    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)

    wave_dir = dataset2wavpath[dataset]
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)


    print("\nExtracting Hubert features...")
    hubert_features = np.zeros((len(datasets), HUBERT_SEQ, HUBERT_DIM), dtype=float)
    audio_paths = [
        os.path.join(wave_dir, "{}.wav".format(utt["Uid"])) for utt in datasets
    ]
    if dataset == "M4Singer":
        audio_paths = [os.path.join(wave_dir, utt["Path"]) for utt in datasets]

    start = 0
    end = 0
    while end < len(audio_paths):
        start = end
        end = start + batch_size
        print("{}/{}...".format(min(len(audio_paths), end), len(audio_paths)))

        hubert_features[start:end] = hubert_encoder(audio_paths[start:end])

    # print(whisper_features.shape)
    output_dir = os.path.join(data_dir, "Hubert")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "{}.pkl".format(dataset_type)), "wb") as f:
        pickle.dump(hubert_features, f)

if __name__ == "__main__":
    print("Loading Model...")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")

    model = model.eval()

    extract_hubert_features("Opencpop", "test")
    extract_hubert_features("Opencpop", "train")
    extract_hubert_features("M4Singer", "test")