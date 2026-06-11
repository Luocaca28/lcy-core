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
from run.eval import eval_MambaJSCC_models
import csv
import os
import gc
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


def _save_train_val_loss_curve(train_records, val_records, log_dir):
    csv_path = os.path.join(log_dir, "train_val_loss_curve.csv")
    png_path = os.path.join(log_dir, "train_val_loss_curve.png")
    val_by_epoch = {row[0]: row[1] for row in val_records}
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, train_loss in train_records:
            writer.writerow([epoch, train_loss, val_by_epoch.get(epoch, "")])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_epochs = [row[0] for row in train_records]
        train_losses = [row[1] for row in train_records]
        val_epochs = [row[0] for row in val_records]
        val_losses = [row[1] for row in val_records]
        plt.figure()
        plt.plot(train_epochs, train_losses, label="train", marker="o", markersize=3)
        if val_records:
            plt.plot(val_epochs, val_losses, label="val", marker="o", markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save train/val loss plot: {exc}")


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


def _check_gate_grad(model, tag):
    model = _unwrap_parallel(model)
    found = False
    for name, p in model.named_parameters():
        if "defscan_gate" not in name:
            continue
        found = True
        grad = None if p.grad is None else p.grad.detach().abs().mean().item()
        val = p.detach().abs().mean().item()
        print(f"[{tag}] {name}: param_abs_mean={val:.6e}, grad_abs_mean={grad}")
    if not found:
        print(f"[{tag}] no defscan_gate params found")


def _format_count(value):
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return str(value)


def _clear_module_hooks(model):
    for module in model.modules():
        for hook_name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        ):
            hooks = getattr(module, hook_name, None)
            if hooks is not None:
                hooks.clear()


class _SNRForward(torch.nn.Module):
    def __init__(self, model, snr):
        super().__init__()
        self.model = model
        self.snr = snr

    def forward(self, x):
        return self.model(x, self.snr)


def _flops_of(model, input_tensor, snr):
    try:
        from fvcore.nn import FlopCountAnalysis

        wrapped = _SNRForward(model, snr)
        flops = FlopCountAnalysis(wrapped, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        unsupported = dict(flops.unsupported_ops())
        del flops, wrapped
        gc.collect()
        if input_tensor.is_cuda:
            torch.cuda.empty_cache()
        return total, unsupported, None
    except Exception as exc:
        gc.collect()
        if input_tensor.is_cuda:
            torch.cuda.empty_cache()
        return None, {}, str(exc)
    finally:
        _clear_module_hooks(model)


def _print_model_profile(config, encoder, decoder, device):
    encoder = _unwrap_parallel(encoder)
    decoder = _unwrap_parallel(decoder)
    snr_list = config.CHANNEL.SNR
    profile_snr = snr_list[0] if isinstance(snr_list, (list, tuple)) else snr_list
    image_size = config.DATA.IMG_SIZE

    was_encoder_training = encoder.training
    was_decoder_training = decoder.training
    encoder.eval()
    decoder.eval()

    input_tensor = None
    feature = None
    decoder_input = None
    try:
        input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            feature = encoder(input_tensor, profile_snr)
            decoder_input = torch.zeros_like(feature)

        enc_flops, _, enc_error = _flops_of(encoder, input_tensor, profile_snr)
        dec_flops, _, dec_error = _flops_of(decoder, decoder_input, profile_snr)
        total_flops = None
        if enc_flops is not None and dec_flops is not None:
            total_flops = enc_flops + dec_flops

        print(f"Total FLOPs: {_format_count(total_flops)}")
        if enc_error:
            print(f"Encoder FLOPs failed: {enc_error}")
        if dec_error:
            print(f"Decoder FLOPs failed: {dec_error}")
    finally:
        del input_tensor, feature, decoder_input
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_encoder_training:
            encoder.train()
        if was_decoder_training:
            decoder.train()

def train_MambaJSCC(config):
    

    distributed, rank, local_rank = _setup_distributed()
    train_loader, test_loader = get_loader(config)
    device = torch.device("cuda", local_rank) if distributed else torch.device("cuda")
    encoder=Mamba_encoder(config).to(device)
    decoder=Mamba_decoder(config).to(device)
    if distributed:
        print(f"Using DDP rank {rank}, local_rank {local_rank}")
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
    channel=Channel(config)
    if _is_main_process():
        _print_model_profile(config, encoder, decoder, device)
    if distributed:
        dist.barrier()

    
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
    val_loss_records = []
    log_dir = _get_log_dir(config)
    eval_fre = getattr(config.TRAIN, "EVAL_FRE", 10)
    debug_gate_grad_fre = getattr(config.TRAIN, "DEBUG_GATE_GRAD_FRE", 0)
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
                global_step = e * len(train_loader) + i
                if (
                    debug_gate_grad_fre > 0
                    and global_step % debug_gate_grad_fre == 0
                    and _is_main_process()
                ):
                    _check_gate_grad(encoder, "encoder")
                    _check_gate_grad(decoder, "decoder")

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
        if eval_fre > 0 and (e + 1) % eval_fre == 0:
            if distributed:
                dist.barrier()
            if _is_main_process():
                print(f"----------validation after epoch {e + 1}----------")
                encoder.eval()
                decoder.eval()
                _, _, _, val_loss = eval_MambaJSCC_models(
                    config,
                    _unwrap_parallel(encoder),
                    _unwrap_parallel(decoder),
                    test_loader=test_loader,
                    save_recon=False,
                    prefix=f"val_epoch_{e + 1:03d}_snr",
                    return_loss=True,
                )
                val_loss_records.append([e + 1, val_loss])
                encoder.train()
                decoder.train()
            if distributed:
                dist.barrier()
    if _is_main_process():
        save_model(_unwrap_parallel(encoder), save_path=config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
        save_model(_unwrap_parallel(decoder), save_path=config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
    if _is_main_process():
        _save_train_val_loss_curve(loss_records, val_loss_records, log_dir)
    if distributed:
        dist.barrier()
