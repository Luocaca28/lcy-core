'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''
from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.datasets import get_loader

import torch.optim as optim
from tqdm import tqdm
import torch
from utils.utils import *
from utils.distortion import *
from torchvision.utils import save_image
from utils.utils import seed_torch
import csv
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _get_log_dir(config):
    log_path = getattr(config.TRAIN, "LOG_PATH", "")
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        return log_path
    base = os.path.commonpath([config.TRAIN.ENCODER_PATH, config.TRAIN.DECODER_PATH])
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _save_loss_curve(records, log_dir):
    csv_path = os.path.join(log_dir, "loss_curve.csv")
    png_path = os.path.join(log_dir, "loss_curve.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        writer.writerows(records)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row[0] for row in records]
        losses = [row[1] for row in records]
        plt.figure()
        plt.plot(epochs, losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save loss plot: {exc}")


def _maybe_data_parallel(model):
    return model


def _unwrap_parallel(model):
    return model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model


def _setup_distributed():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return False, 0, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), local_rank


def _is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _reduce_scalar(value, device):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor.item()

def train_MambaJSCC(config):
    

    distributed, rank, local_rank = _setup_distributed()
    train_loader, _ = get_loader(config)
    device = torch.device("cuda", local_rank) if distributed else torch.device("cuda")
    encoder=Mamba_encoder(config).to(device)
    decoder=Mamba_decoder(config).to(device)
    if distributed:
        print(f"Using DDP rank {rank}, local_rank {local_rank}")
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
    channel=Channel(config)

    
    optimizer_encoder = optim.AdamW(encoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)
    optimizer_decoder = optim.AdamW(decoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)

    cosineScheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_encoder, T_max=config.TRAIN.EPOCHS, eta_min=0, last_epoch=-1)
    warmUpScheduler_encoder = GradualWarmupScheduler(
        optimizer=optimizer_encoder, multiplier=2., warm_epoch=0.1,  # CHDDIM_config.epoch // 10,
        after_scheduler=cosineScheduler_encoder)
    
    cosineScheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer_decoder, T_max=config.TRAIN.EPOCHS, eta_min=0, last_epoch=-1)
    warmUpScheduler_decoder = GradualWarmupScheduler(
        optimizer=optimizer_decoder, multiplier=2., warm_epoch=0.1,  # CHDDIM_config.epoch // 10,
        after_scheduler=cosineScheduler_decoder)
    
    criterion=loss_matrix(config)

    matrix=eval_matrix(config) 

    encoder.train()
    decoder.train()
    
    if _is_main_process():
        print(config.MODEL.VSSM.EMBED_DIM, config.MODEL.VSSM.DEPTHS)
        print("----------training: ls:128---OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}-------".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE))
    #print("---training---, --- ")
    seed_torch()
    loss_records = []
    log_dir = _get_log_dir(config)
    for e in range(config.TRAIN.EPOCHS):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(e)
        loss_ave=0
        
        with tqdm(train_loader, dynamic_ncols=False, disable=not _is_main_process()) as tqdmTrainData:
            for i, (input_image, target) in enumerate(tqdmTrainData):
                #save_image(input_image,"/home/wutong/code/ManbaJSCC/{}.png".format(i))
                SNR_list=config.CHANNEL.SNR
                SNR_index=torch.randint(0,len(SNR_list),(1,)).item()


                SNR=SNR_list[SNR_index]
                #-----------------encoder---------------------
                input_image = input_image.to(device, non_blocking=True)
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()     

                feature = encoder(input_image, SNR)
                CBR=feature.numel()/input_image.numel()/2
                
                #----------------channel---------------------
                received, pwr, h = channel.forward(feature, SNR)
                if config.CHANNEL.TYPE=='rayleigh':
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                    
                elif config.CHANNEL.TYPE=='awgn':
                    pass
                else:
                    raise ValueError("channel type error")
                #-----------------decoder---------------------
                received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                recon_image = decoder(received, SNR)
                

                
                loss = criterion(recon_image, input_image, feature,opt_idx=0, global_step=e)
                loss.backward()

                performance=matrix(recon_image, input_image)
                
                
                loss_ave=(loss_ave+loss.item())

                torch.nn.utils.clip_grad_norm_(    
                    encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), 1)
                
                optimizer_encoder.step()
                optimizer_decoder.step()

                tqdmTrainData.set_postfix({
                    'e':e,
                    'loss': (loss.item(),loss_ave/(i+1)),
                    'matrix':performance,
                    'CBR':CBR,
                    'SNR':SNR,
                    "LR": (optimizer_encoder.state_dict()['param_groups'][0]["lr"],optimizer_encoder.state_dict()['param_groups'][0]["lr"])
                }
                    )

        warmUpScheduler_encoder.step()
        warmUpScheduler_decoder.step()
        loss_ave=loss_ave/(i+1)
        loss_ave = _reduce_scalar(loss_ave, device)
        if _is_main_process():
            loss_records.append([e + 1, loss_ave])
        if _is_main_process() and (e + 1) % (config.TRAIN.SAVE_FRE) == 0:
            # save_model(encoder, save_path=config.TRAIN.ENCODER_PATH + "ls32_OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            # save_model(decoder, save_path=config.TRAIN.DECODER_PATH + "ls32_OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            save_model(_unwrap_parallel(encoder), save_path=config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            save_model(_unwrap_parallel(decoder), save_path=config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
    if _is_main_process():
        _save_loss_curve(loss_records, log_dir)
    if distributed:
        dist.barrier()
