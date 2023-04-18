import json
import itertools
import logging
import datetime
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import time
import pickle
import psutil
import sys
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp

from utils.utils import pad_or_trim, AttrDict, get_logger, save_checkpoint
from utils.hifigan_utils import plot_spectrogram

from models.encoder import LF0Encoder, PPGEncoder, PhonemeEncoder
from models.decoder import MelDecoder
from models.svc import DIFFSVC
from models.vocoder.diffwave_models import DiffWave

from datasets.dataset_gan import mel_spectrogram
from datasets.dataset_diff import SVCDIFFDataset

from config import (
    data_path, root_path,
    TARGET_SAMPLE_RATE, MAX_MEL_LENGTH,  # 44100, 1000 
    WHISPER_DIM, HUBERT_DIM, MCEP_DIM,    # 1024, 1024, 80
    MAX_AUDIO_LENGTH                       # 256000
)


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
    
def cleanup():
    dist.destroy_process_group()
    
def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def evaluate(val_loader, svc, sw, steps, h, device):
    svc.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            mel_gt, wav_gt, mel_mask, wav_mask, lf0, ppg, pho = batch
            mel_gt = mel_gt.to(device)
            mel_mask = mel_mask.to(device)
            wav_mask = wav_mask.to(device)
            wav_gt = wav_gt.to(device)
            mel_gt = mel_gt.to(device)
            
            
            wav_pred = svc.inference(lf0.unsqueeze(1).to(device),ppg.to(device),pho.to(device), 
                                fast_sampling=True)
            
            val_err_tot += F.l1_loss(wav_pred * wav_mask, wav_gt * wav_mask).item()

            if j <= 4:
                if steps == 0:
                    sw.add_audio('gt/y_{}'.format(j), wav_gt[0], steps, TARGET_SAMPLE_RATE)
                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(mel_gt.squeeze(0).cpu().numpy(), title=f"GT {j}"), steps)

                sw.add_audio('generated/y_hat_{}'.format(j), wav_pred[0], steps, TARGET_SAMPLE_RATE)
                mel_pred = mel_spectrogram(wav_pred, h.n_fft, h.n_mels, TARGET_SAMPLE_RATE,
                                          hop_size=h.hop_samples, win_size=h.hop_samples*4,
                                          fmin=20, fmax=TARGET_SAMPLE_RATE / 2.0)
                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                              plot_spectrogram(mel_pred.squeeze(0).cpu().numpy(), title=f"GEN {j}"), steps)

        val_err = val_err_tot / (j+1)
        sw.add_scalar("validation/mel_spec_error", val_err, steps)

    svc.train()
    return val_err

def train(rank, world_size, experiment_path, logger, args, h):
    
    ## parameters settings !!
    num_epoches = 100
    lf0_out_dims = 8
    ppg_out_dims = 128
    pho_out_dims = 128
    batch_size = 2
    num_workers = 4
    
    
    learning_rate = h.learning_rate
    lr_decay=       0.999
    
    
    save_path = os.path.join(experiment_path, "models")
    os.makedirs(save_path, exist_ok=True)

    tb_path = os.path.join(experiment_path, "tb_dir")
    os.makedirs(tb_path, exist_ok=True)
    
    device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')
    
    lf0_encoder = LF0Encoder(in_dims=1,out_dims=lf0_out_dims).to(device)
    ppg_encoder = PPGEncoder(output_dims=ppg_out_dims, d_model=WHISPER_DIM).to(device)
    pho_encoder = PhonemeEncoder(output_dims=pho_out_dims, d_model=HUBERT_DIM).to(device)
    
    
    mel_decoder = MelDecoder(
                    lf0_dims=lf0_out_dims,ppg_dims=ppg_out_dims,
                    phoneme_dims=pho_out_dims,output_dims=MCEP_DIM).to(device)
    
    
    
    diffwave = DiffWave(h).to(device)
    
    
    if rank == 0:
        sw = SummaryWriter(os.path.join(tb_path, 'logs'))
        
        if device.type == 'cuda':
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
        else:
            logger.info('WARNING: Using device {}'.format(device))
    
    ## setup dataset and loader
    trainset = SVCDIFFDataset(
                dataset="Opencpop",
                dataset_type="train", args=None,
                n_fft=h.n_fft, num_mels=h.n_mels,hop_size=h.hop_samples, 
                win_size=h.hop_samples*4, sampling_rate=TARGET_SAMPLE_RATE,  
                fmin=20, fmax=TARGET_SAMPLE_RATE / 2.0, 
                mel_crop_length=MAX_MEL_LENGTH,
                audio_crop_length=MAX_AUDIO_LENGTH
            )
    
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    
    valset = SVCDIFFDataset(
                dataset="Opencpop",
                dataset_type="test", args=None,
                n_fft=h.n_fft, num_mels=h.n_mels,hop_size=h.hop_samples, 
                win_size=h.hop_samples*4, sampling_rate=TARGET_SAMPLE_RATE,  
                fmin=20, fmax=TARGET_SAMPLE_RATE / 2.0, mel_crop_length=MAX_MEL_LENGTH,
                audio_crop_length=MAX_AUDIO_LENGTH
            )
    
    train_loader = DataLoader(
                  trainset, 
                  num_workers=num_workers, 
                  shuffle=False if world_size > 1 else True,
                  sampler = train_sampler,
                  batch_size=batch_size,
                  pin_memory=True,
                  drop_last=True
                )

    val_loader = DataLoader(
                      valset, 
                      num_workers=num_workers, 
                      shuffle= False,
                      batch_size=1,
                      pin_memory=True,
                      drop_last=True
                    )
    
    ## setup generator and discriminator
    svc = DIFFSVC(lf0_encoder,ppg_encoder,pho_encoder,decoder=mel_decoder,vocoder=diffwave,
             noise_schedule=h.noise_schedule, 
              inference_noise_schedule=h.inference_noise_schedule,
             audio_len=MAX_AUDIO_LENGTH).to(device)
    
    if world_size > 1:
        svc = DDP(svc, device_ids=[rank])
    
    ## setup optimizer and scheduler
    
    optim_g = torch.optim.AdamW(svc.parameters(), learning_rate)
    
    # last_epoch = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
    
    
    svc.train()
    
    ## start training !!
    steps = 0
    # Best results on validation dataset
    best_val_result = np.inf
    best_val_epoch = -1
    
    if rank == 0:
        _ = evaluate(val_loader,svc, sw, steps=steps,h=h,device=device)
        
    ## prepare training noise scheduling
    beta = np.array(h.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32)).to(device)

    for epoch in range(num_epoches):
        
        if world_size > 1:
            train_sampler.set_epoch(epoch)
            
        for i, batch in enumerate(train_loader):

            start_b = time.time()

            mel_gt, wav_gt, mel_mask, wav_mask, lf0, ppg, pho = batch
            mel_gt = mel_gt.to(device)
            mel_mask = mel_mask.to(device)
            wav_mask = wav_mask.to(device)
            wav_gt = wav_gt.to(device)
            mel_gt = mel_gt.to(device)

            N, T = wav_gt.shape
            t = torch.randint(0, len(h.noise_schedule), [N], device=wav_gt.device)

            # generate noise input at t 
            noise_scale = noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(wav_gt)
            noisy_wav = noise_scale_sqrt * wav_gt + (1.0 - noise_scale)**0.5 * noise

            # Generator
            optim_g.zero_grad()
            predicted_noise, mel_rec = svc(lf0.unsqueeze(1).to(device),
                                     ppg.to(device),pho.to(device), t, 
                                     noisy_wav.to(device))

            loss = F.l1_loss(predicted_noise.squeeze(1) * wav_mask, 
                                     noise * wav_mask) * 10
            loss_mel = F.l1_loss(mel_rec * mel_mask, mel_gt * mel_mask)

            loss_total = loss + loss_mel
            loss_total.backward()
            optim_g.step()
            
            # Sum memory usage across devices.
            mem = torch.tensor(memory_usage_psutil()).float().to(device)
            if world_size > 1:
                dist.all_reduce(mem, op=dist.ReduceOp.SUM)
            if rank == 0:
                if steps % 50 == 0:

                    with torch.no_grad():
                        loss = F.l1_loss(predicted_noise.squeeze(1) * wav_mask, 
                                         noise * wav_mask).item()
                        loss_mel = F.l1_loss(mel_rec * mel_mask, mel_gt * mel_mask).item()
                        total_loss = loss * 10 + loss_mel
                        time_past = time.time() - start_b
                        logging.info(
                            f" Epoch : {epoch:03d} | Steps : {steps:06d} | "
                            f"Gen Loss Total : {total_loss:4.4f} | "
                            f"Mel-Spec. Error : {loss_mel:4.4f} | s/b : {time_past:4.4f}"
                        )

                    sw.add_scalar("training/total_loss", total_loss, steps)


            del loss_total
            steps += 1

        scheduler_g.step()

        if rank == 0:
            val_result = evaluate(val_loader,svc, sw, steps=steps,h=h,device=device)    
            # Save model
            if val_result <= best_val_result:
                # model file
                save_checkpoint(os.path.join(save_path,f"g_{epoch:03d}.pt"),
                    {'svc': (svc.module if world_size > 1 else svc).state_dict()})

                if best_val_epoch != -1:
                    os.system(
                        "rm {}".format(
                            os.path.join(save_path,f"g_{best_val_epoch:03d}.pt")
                        )
                    )

                best_val_result = val_result
                best_val_epoch = epoch

            logging.info(
                f"Epoch : {epoch:03d} | Best Epoch : {best_val_epoch:03d} | Test : {val_result:4.4f}" 
            )
            
            
def main(rank, world_size, savepath,args, h):
    
    port = 5554

    setup(rank, world_size, port)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    logger = get_logger(os.path.join(savepath,"terminal.log"))

    try:
        train(rank, world_size, savepath, logger, args, h)
    except:
        import traceback
        logging.error(traceback.format_exc())
        raise

    cleanup()
    
if __name__ == "__main__":
    
    from config import parser
    
    args = parser.parse_args()
    
    # GPU counts
    world_size = torch.cuda.device_count()
    
    # setup experiment file dir
    experiment_path = os.path.join(root_path, "experiment")
    experiment_name = "DIFFWAVE"
    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_name += f"_{experiment_id}"
    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    logger = get_logger(os.path.join(experiment_path,"terminal.log"))
    ## loading HIFI-GAN LJ_FT_T2_V3 generator-V3 model 
    # trained on dataset (LJSpeech) fined tuned by Tacotron2 Pretained model
    h = AttrDict(
        # Training params
        learning_rate=2e-4,
        max_grad_norm=None,

        # Data params
        sample_rate=22050,
        n_mels=80,
        n_fft=1024,
        hop_samples=256,
   
        # Model params
        residual_layers=30,
        residual_channels=64,
        dilation_cycle_length=10,
        unconditional = False,
        noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    )
    
    
    try:
        # mp.set_start_method("forkserver")
        if world_size > 1:
            mp.spawn(
                main,
                args=(world_size, experiment_path, args, h),
                nprocs=world_size,
                join=True
            )
        else:
            train(0, world_size, experiment_path, logger, args, h)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)