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
from datetime import timedelta
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


def _save_loss_curve(train_records, val_records, log_dir):
    csv_path = os.path.join(log_dir, "loss_curve.csv")
    png_path = os.path.join(log_dir, "loss_curve.png")
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
        plt.plot(train_epochs, train_losses, marker="o", label="train")
        if val_records:
            plt.plot(val_epochs, val_losses, marker="s", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save loss plot: {exc}")


def _collect_tri_path_weight_stats(model, prefix):
    diag_values = []
    offdiag_values = []
    abs_values = []
    base_abs_values = []
    for name, module in _unwrap_parallel(model).named_modules():
        if "tri_merge" not in name or not hasattr(module, "proj"):
            continue
        weight = module.proj.weight.detach()
        out_channels, in_channels, kernel = weight.shape
        if kernel != 1 or in_channels != 3 * out_channels:
            continue
        c = out_channels
        w0 = weight[:, 0 * c : 1 * c, 0]
        w1 = weight[:, 1 * c : 2 * c, 0]
        wd = weight[:, 2 * c : 3 * c, 0]
        wd_diag = torch.diag(wd).mean()
        wd_offdiag = (wd - torch.diag(torch.diag(wd))).abs().mean()
        wd_abs = wd.abs().mean()
        base_abs = 0.5 * (w0.abs().mean() + w1.abs().mean())
        diag_values.append(wd_diag)
        offdiag_values.append(wd_offdiag)
        abs_values.append(wd_abs)
        base_abs_values.append(base_abs)

    if not abs_values:
        return {
            f"{prefix}_wd_diag_mean": "",
            f"{prefix}_wd_offdiag_abs_mean": "",
            f"{prefix}_wd_abs_mean": "",
            f"{prefix}_wd_to_base_abs_ratio": "",
        }

    wd_diag_mean = torch.stack(diag_values).mean().item()
    wd_offdiag_mean = torch.stack(offdiag_values).mean().item()
    wd_abs_mean = torch.stack(abs_values).mean().item()
    base_abs_mean = torch.stack(base_abs_values).mean().item()
    return {
        f"{prefix}_wd_diag_mean": wd_diag_mean,
        f"{prefix}_wd_offdiag_abs_mean": wd_offdiag_mean,
        f"{prefix}_wd_abs_mean": wd_abs_mean,
        f"{prefix}_wd_to_base_abs_ratio": wd_abs_mean / (base_abs_mean + 1e-12),
    }


def _save_tri_path_weight_curve(records, log_dir):
    if not records:
        return
    csv_path = os.path.join(log_dir, "tri_path_weight_curve.csv")
    png_path = os.path.join(log_dir, "tri_path_weight_curve.png")
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in records]
        plt.figure()
        for key in fieldnames:
            if key == "epoch":
                continue
            values = [row[key] for row in records]
            if any(value == "" for value in values):
                continue
            plt.plot(epochs, values, marker="o", label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Tri-path weight statistic")
        plt.title("Third Path Weight Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save tri-path weight plot: {exc}")


def _maybe_data_parallel(model):
    return model


def _unwrap_parallel(model):
    return model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model


def _split_tri_merge_params(model):
    base_params = []
    tri_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tri_merge" in name:
            tri_params.append(param)
        else:
            base_params.append(param)
    return base_params, tri_params


def _build_adamw_with_tri_merge(model, base_lr, weight_decay, tri_lr_mult):
    base_params, tri_params = _split_tri_merge_params(model)
    groups = [{"params": base_params, "lr": base_lr, "weight_decay": weight_decay}]
    if tri_params:
        groups.append(
            {
                "params": tri_params,
                "lr": base_lr * tri_lr_mult,
                "weight_decay": 0.0,
            }
        )
    return optim.AdamW(groups)


def _print_optimizer_groups(name, optimizer):
    for idx, group in enumerate(optimizer.param_groups):
        print(
            f"{name} group {idx}: lr={group['lr']} "
            f"wd={group.get('weight_decay', 0.0)} n_params={len(group['params'])}"
        )


def _inspect_tri_merge_grad(model, tag):
    for name, param in model.named_parameters():
        if "tri_merge" not in name:
            continue
        grad = None if param.grad is None else param.grad.detach().abs().mean().item()
        print(f"[{tag}] {name}: grad_mean={grad}")


def _setup_distributed():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return False, 0, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
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


def _format_count_with_commas(value):
    return "N/A" if value is None else f"{value:,}"


def _count_params(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _param_memory_mb(model):
    return sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)


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
        enc_params, _ = _count_params(encoder)
        dec_params, _ = _count_params(decoder)
        total_params = enc_params + dec_params
        enc_param_mem = _param_memory_mb(encoder)
        dec_param_mem = _param_memory_mb(decoder)

        input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            feature = encoder(input_tensor, profile_snr)
            decoder_input = torch.zeros_like(feature)

        enc_flops, _, enc_error = _flops_of(encoder, input_tensor, profile_snr)
        dec_flops, _, dec_error = _flops_of(decoder, decoder_input, profile_snr)
        total_flops = None
        if enc_flops is not None and dec_flops is not None:
            total_flops = enc_flops + dec_flops

        print(f"Encoder params: {_format_count(enc_params)} ({_format_count_with_commas(enc_params)})")
        print(f"Decoder params: {_format_count(dec_params)} ({_format_count_with_commas(dec_params)})")
        print(f"Total params:   {_format_count(total_params)} ({_format_count_with_commas(total_params)})")
        print(f"Encoder size: {enc_param_mem:.2f} MB")
        print(f"Decoder size: {dec_param_mem:.2f} MB")
        print(f"Total size:   {enc_param_mem + dec_param_mem:.2f} MB")
        print(f"Encoder FLOPs: {_format_count(enc_flops)}")
        print(f"Decoder FLOPs: {_format_count(dec_flops)}")
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
    val_data_dir = getattr(config.DATA, "val_data_dir", config.DATA.test_data_dir)
    if _is_main_process():
        print(f"Validation data dir: {val_data_dir}")
    train_loader, val_loader = get_loader(config, test_data_dir=val_data_dir)
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

    
    tri_lr_mult = getattr(config.MODEL.VSSM, "TRI_MERGE_LR_MULT", 1.0)
    optimizer_encoder = _build_adamw_with_tri_merge(
        encoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
    )
    optimizer_decoder = _build_adamw_with_tri_merge(
        decoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
    )
    if _is_main_process():
        _print_optimizer_groups("encoder", optimizer_encoder)
        _print_optimizer_groups("decoder", optimizer_decoder)

    total_steps = max(1, config.TRAIN.EPOCHS * len(train_loader))
    warmup_steps = max(1, int(config.TRAIN.WARMUP_EPOCHS * len(train_loader)))
    cosine_steps = max(1, total_steps - warmup_steps)
    cosineScheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_encoder, T_max=cosine_steps, eta_min=0, last_epoch=-1)
    warmUpScheduler_encoder = GradualWarmupScheduler(
        optimizer=optimizer_encoder, multiplier=2., warm_epoch=warmup_steps,
        after_scheduler=cosineScheduler_encoder)
    
    cosineScheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_decoder, T_max=cosine_steps, eta_min=0, last_epoch=-1)
    warmUpScheduler_decoder = GradualWarmupScheduler(
        optimizer=optimizer_decoder, multiplier=2., warm_epoch=warmup_steps,
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
    tri_weight_records = []
    log_dir = _get_log_dir(config)
    eval_fre = getattr(config.TRAIN, "EVAL_FRE", 10)
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
                if getattr(config.MODEL.VSSM, "TRI_MERGE_DEBUG", False) and _is_main_process() and i == 0:
                    _inspect_tri_merge_grad(encoder, f"epoch {e} encoder")
                    _inspect_tri_merge_grad(decoder, f"epoch {e} decoder")

                performance=matrix(recon_image, input_image)
                
                
                loss_ave=(loss_ave+loss.item())

                torch.nn.utils.clip_grad_norm_(    
                    encoder.parameters(), config.TRAIN.CLIP_GRAD)
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), config.TRAIN.CLIP_GRAD)
                
                optimizer_encoder.step()
                optimizer_decoder.step()
                warmUpScheduler_encoder.step()
                warmUpScheduler_decoder.step()

                tqdmTrainData.set_postfix({
                    'e':e,
                    'loss': (loss.item(),loss_ave/(i+1)),
                    'matrix':performance,
                    'CBR':CBR,
                    'SNR':SNR,
                    "LR": tuple(group["lr"] for group in optimizer_encoder.param_groups)
                    }
                    )

        loss_ave=loss_ave/(i+1)
        loss_ave = _reduce_scalar(loss_ave, device)
        if _is_main_process():
            loss_records.append([e + 1, loss_ave])
            tri_record = {"epoch": e + 1}
            tri_record.update(_collect_tri_path_weight_stats(encoder, "encoder"))
            tri_record.update(_collect_tri_path_weight_stats(decoder, "decoder"))
            tri_weight_records.append(tri_record)
        if _is_main_process() and (e + 1) % (config.TRAIN.SAVE_FRE) == 0:
            # save_model(encoder, save_path=config.TRAIN.ENCODER_PATH + "ls32_OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            # save_model(decoder, save_path=config.TRAIN.DECODER_PATH + "ls32_OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            save_model(_unwrap_parallel(encoder), save_path=config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
            save_model(_unwrap_parallel(decoder), save_path=config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
        run_validation = eval_fre > 0 and (
            (e + 1) % eval_fre == 0 or (e + 1) == config.TRAIN.EPOCHS
        )
        if run_validation:
            if distributed:
                dist.barrier()
            if _is_main_process():
                print(f"----------validation after epoch {e + 1}----------")
                encoder.eval()
                decoder.eval()
                save_final_val_curves = (e + 1) == config.TRAIN.EPOCHS
                _, _, _, val_loss = eval_MambaJSCC_models(
                    config,
                    _unwrap_parallel(encoder),
                    _unwrap_parallel(decoder),
                    test_loader=val_loader,
                    save_recon=False,
                    save_curves=save_final_val_curves,
                    prefix=f"val_epoch_{e + 1:03d}_snr",
                    criterion=criterion,
                    global_step=e,
                )
                if val_loss is not None:
                    val_loss_records.append([e + 1, val_loss])
                encoder.train()
                decoder.train()
            if distributed:
                dist.barrier()
    if _is_main_process():
        save_model(_unwrap_parallel(encoder), save_path=config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
        save_model(_unwrap_parallel(decoder), save_path=config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt')
    if _is_main_process():
        _save_loss_curve(loss_records, val_loss_records, log_dir)
        _save_tri_path_weight_curve(tri_weight_records, log_dir)
    if distributed:
        dist.barrier()
