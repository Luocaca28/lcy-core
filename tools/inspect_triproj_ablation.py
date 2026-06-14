import argparse
import os
import sys
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_config_args(project_path, config_name):
    return SimpleNamespace(
        config_name=config_name,
        project_path=project_path,
        model_config_path=os.path.join(
            project_path, "configs", "vssm", f"vssm_tiny_{config_name}.yaml"
        ),
        train_config_path=os.path.join(
            project_path, "configs", "train", f"vssm_tiny_{config_name}.yaml"
        ),
    )


def checkpoint_path(config, kind):
    base = config.TRAIN.ENCODER_PATH if kind == "encoder" else config.TRAIN.DECODER_PATH
    return (
        base
        + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
            config.MODEL.VSSM.OUT_CHANS,
            config.MODEL.VSSM.Extent,
            config.TRAIN.LOSS,
            config.MODEL.VSSM.SCAN_NUMBER,
            config.CHANNEL.SNR,
            config.CHANNEL.ADAPTIVE,
            config.CHANNEL.TYPE,
            len(config.MODEL.VSSM.EMBED_DIM),
            config.MODEL.VSSM.EMBED_DIM,
            config.MODEL.VSSM.DEPTHS,
            config.DATA.IMG_SIZE,
        )
        + ".pt"
    )


def _eye_like(block, value):
    eye = torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
    return value * eye


def apply_ablation(model, mode, def_init):
    if mode == "normal":
        return
    for name, module in model.named_modules():
        if "tri_merge" not in name or not hasattr(module, "proj"):
            continue
        weight = module.proj.weight
        out_channels, in_channels, kernel = weight.shape
        if kernel != 1 or in_channels != 3 * out_channels:
            continue
        with torch.no_grad():
            c = out_channels
            wd = weight[:, 2 * c : 3 * c, 0]
            if mode == "wd_zero":
                wd.zero_()
            elif mode == "wd_init":
                wd.copy_(_eye_like(wd, def_init))
            elif mode == "wd_diag":
                wd.copy_(torch.diag(torch.diag(wd)))
            else:
                raise ValueError(f"unknown ablation mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="DIV2K_defscan")
    parser.add_argument("--project_path", default=os.getcwd())
    parser.add_argument(
        "--mode",
        default="normal",
        choices=["normal", "wd_zero", "wd_init", "wd_diag"],
    )
    parser.add_argument("--encoder_path", default=None)
    parser.add_argument("--decoder_path", default=None)
    parser.add_argument("--save_recon", action="store_true")
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    sys.path.insert(0, project_path)
    os.chdir(project_path)

    from configs.config import get_config
    from run.eval import eval_MambaJSCC_models

    config = get_config(build_config_args(project_path, args.config_name))
    encoder_path = args.encoder_path or checkpoint_path(config, "encoder")
    decoder_path = args.decoder_path or checkpoint_path(config, "decoder")
    encoder = torch.load(encoder_path, weights_only=False)
    decoder = torch.load(decoder_path, weights_only=False)

    def_init = getattr(config.MODEL.VSSM, "DEFSCAN_DEF_INIT", 0.05)
    apply_ablation(encoder, args.mode, def_init)
    apply_ablation(decoder, args.mode, def_init)
    eval_MambaJSCC_models(
        config,
        encoder,
        decoder,
        save_recon=args.save_recon,
        prefix=f"triproj_{args.mode}",
    )


if __name__ == "__main__":
    main()
