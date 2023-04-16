import json
import itertools
import logging
import datetime
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from models.svc import SVC
from models.vocoder.hifigan_models import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, 
    feature_loss, generator_loss, discriminator_loss
)

from datasets.dataset_gan import SVCGANDataset, mel_spectrogram

from config import (
    data_path, root_path,
    TARGET_SAMPLE_RATE, MAX_MEL_LENGTH,  # 44100, 1000 
    WHISPER_DIM, HUBERT_DIM, MCEP_DIM   # 1024, 1024, 80
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
            mel_gt, y, y_mel,y_mel_mask, y_mask, lf0, ppg, pho = batch
            
            y_mel = y_mel.to(device)
            y_mel_mask = y_mel_mask.to(device)
            y = y.unsqueeze(1).to(device)
            
            y_g_hat, mel_rec = svc(lf0.unsqueeze(1).to(device),ppg.to(device),pho.to(device))
            
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                          h.hop_size, h.win_size,
                                          h.fmin, h.fmax_loss)
            
            val_err_tot += F.l1_loss(y_mel * y_mel_mask, y_g_hat_mel * y_mel_mask).item()

            if j <= 4:
                if steps == 0:
                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, TARGET_SAMPLE_RATE)
                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(mel_gt[0], title=f"GT {j}"), steps)

                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, TARGET_SAMPLE_RATE)
                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                             h.sampling_rate, h.hop_size, h.win_size,
                                             h.fmin, h.fmax)
                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy(), title=f"HIFI GEN {j}"), steps)

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
    
    learning_rate = 1e-4
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
    
    
    
    generator = Generator(h).to(device)
    generator.load_state_dict(torch.load("weights/hifigan_generator")['generator'])
    
    
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
    trainset = SVCGANDataset(
                    dataset="Opencpop",
                    dataset_type="train", args=args,
                n_fft=h.n_fft, num_mels=h.num_mels,hop_size=h.hop_size, 
                win_size=h.win_size, sampling_rate=TARGET_SAMPLE_RATE,  
        fmin=h.fmin, fmax=h.fmax, fmax_loss=h.fmax_loss, mel_crop_length=MAX_MEL_LENGTH)
    
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    
    valset = SVCGANDataset(
                    dataset="Opencpop",
                    dataset_type="test", args=args,
                n_fft=h.n_fft, num_mels=h.num_mels,hop_size=h.hop_size, 
                win_size=h.win_size, sampling_rate=TARGET_SAMPLE_RATE,  
        fmin=h.fmin, fmax=h.fmax, fmax_loss=h.fmax_loss, mel_crop_length=MAX_MEL_LENGTH)
    
    train_loader = DataLoader(
                  trainset, 
                  num_workers=h.num_workers, 
                  shuffle=False if world_size > 1 else True,
                  sampler = train_sampler,
                  batch_size=batch_size,
                  pin_memory=True,
                  drop_last=True
                )

    val_loader = DataLoader(
                      valset, 
                      num_workers=h.num_workers, 
                      shuffle= False,
                      batch_size=1,
                      pin_memory=True,
                      drop_last=True
                    )
    
    ## setup generator and discriminator
    svc = SVC(lf0_encoder,ppg_encoder,pho_encoder,decoder=mel_decoder,vocoder=generator).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    if world_size > 1:
        svc = DDP(svc, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank])
        msd = DDP(msd, device_ids=[rank])
    
    ## setup optimizer and scheduler
    
    optim_g = torch.optim.AdamW(svc.parameters(), learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                learning_rate, betas=[h.adam_b1, h.adam_b2])
    
    # last_epoch = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)
    
    svc.train()
    mpd.train()
    msd.train()
    
    ## start training !!
    steps = 0
    # Best results on validation dataset
    best_val_result = -np.inf
    best_val_epoch = -1
    
    if rank == 0:
        best_val_result = evaluate(val_loader,svc, sw, steps=steps,h=h,device=device)

    for epoch in range(num_epoches):
        
        if world_size > 1:
            train_sampler.set_epoch(epoch)
            
        for i, batch in enumerate(train_loader):

            start_b = time.time()

            mel_gt, y, y_mel,y_mel_mask, y_mask, lf0, ppg, pho = batch

            y_mel = y_mel.to(device)
            y_mel_mask = y_mel_mask.to(device)
            y = y.unsqueeze(1).to(device)
            mel_gt = mel_gt.to(device)

            y_g_hat, mel_rec = svc(lf0.unsqueeze(1).to(device),ppg.to(device),pho.to(device))


            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, TARGET_SAMPLE_RATE, 
                                        h.hop_size, h.win_size,
                                        h.fmin, h.fmax_loss)
            # MPD
            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel * y_mel_mask, y_g_hat_mel * y_mel_mask) * 45
            loss_mel_mid = F.l1_loss(mel_gt * y_mel_mask, mel_rec * y_mel_mask) * 20

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_mel_mid

            # print(torch.autograd.grad(loss_gen_all, svc.parameters()))

            loss_gen_all.backward()
            optim_g.step()
            
            # Sum memory usage across devices.
            mem = torch.tensor(memory_usage_psutil()).float().to(device)
            if world_size > 1:
                dist.all_reduce(mem, op=dist.ReduceOp.SUM)
            if rank == 0:
                if steps % 50 == 0:



                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel * y_mel_mask, y_g_hat_mel * y_mel_mask).item()

                        time_past = time.time() - start_b
                        logging.info(
                            f" Epoch : {epoch:03d} | Steps : {steps:06d} | "
                            f"Gen Loss Total : {loss_gen_all.item():4.4f} | "
                            f"Dis Loss Total : {loss_disc_all.item():4.4f} | "
                            f"Mel-Spec. Error : {mel_error:4.4f} | s/b : {time_past:4.4f} ｜ "
                            f"Mem Usage : {mem.item():.2f} MB"
                        )

                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)


            del loss_gen_all
            del loss_disc_all 
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        if rank == 0:
            val_result = evaluate(val_loader,svc, sw, steps=steps,h=h,device=device)    
            # Save model
            if val_result >= best_val_result:
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
    experiment_name = "HIFI_GAN"
    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_name += f"_{experiment_id}"
    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    logger = get_logger(os.path.join(experiment_path,"terminal.log"))
    ## loading HIFI-GAN LJ_FT_T2_V3 generator-V3 model 
    # trained on dataset (LJSpeech) fined tuned by Tacotron2 Pretained model
    with open("configs/hifigan_config.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    
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