import argparse
import os
import sys
from types import SimpleNamespace


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


def check_config(project_path, config_name):
    from configs.config import get_config

    config = get_config(build_config_args(project_path, config_name))
    print("SCAN_NUMBER:", config.MODEL.VSSM.SCAN_NUMBER)
    print("USE_DEFSCAN:", config.MODEL.VSSM.USE_DEFSCAN)
    print("DEFSCAN_SCALE:", config.MODEL.VSSM.DEFSCAN_SCALE)

    if config_name.endswith("defscan"):
        assert config.MODEL.VSSM.SCAN_NUMBER == 2
        assert config.MODEL.VSSM.USE_DEFSCAN is True
        assert config.MODEL.VSSM.DEFSCAN_SCALE == "preserve"
    return config


def check_defmamba_branch(device):
    import torch
    from models.defmamba_scan import DeformableLayer, DeformableLayerReverse

    B, C, H, W = 2, 64, 16, 16
    L = H * W
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    DS = DeformableLayer(index=0, embed_dim=C, debug=False).to(device)
    DR = DeformableLayerReverse().to(device)

    deform_tokens, indices = DS(x)
    print("deform_tokens:", tuple(deform_tokens.shape))
    print(
        "indices:",
        tuple(indices.shape),
        int(indices.min().item()),
        int(indices.max().item()),
    )

    assert deform_tokens.shape == (B, L, C)
    assert indices.shape == (B, L)
    assert indices.min().item() >= 0
    assert indices.max().item() < L

    y_def = torch.randn(B, C, L, device=device, requires_grad=True)
    restored = DR(y_def, indices)
    print("restored:", tuple(restored.shape))
    assert restored.shape == (B, C, L)

    loss = deform_tokens.mean() + restored.mean()
    loss.backward()
    print("DefMamba branch forward/backward ok")


def check_model_forward(config, device, image_size, snr, full_channel):
    import torch
    from models.network import Mamba_encoder, Mamba_decoder

    encoder = Mamba_encoder(config).to(device)
    decoder = Mamba_decoder(config).to(device)
    encoder.train()
    decoder.train()

    x = torch.randn(1, 3, image_size, image_size, device=device)
    feature = encoder(x, snr)
    print("feature:", tuple(feature.shape))

    cbr = feature.numel() / x.numel() / 2
    print("CBR:", cbr)

    if full_channel:
        from models.channel import Channel

        channel = Channel(config)
        received, pwr, h = channel.forward(feature, snr)
        if config.CHANNEL.TYPE == "rayleigh":
            sigma_square = 1.0 / (10 ** (snr / 10))
            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
        elif config.CHANNEL.TYPE != "awgn":
            raise ValueError("channel type error")
        decoder_input = torch.cat((torch.real(received), torch.imag(received)), dim=2)
        decoder_input = decoder_input * torch.sqrt(pwr)
    else:
        decoder_input = feature

    recon = decoder(decoder_input, snr)
    print("recon:", tuple(recon.shape))
    assert recon.shape == x.shape

    loss = (recon - x).pow(2).mean()
    loss.backward()
    print("model forward/backward ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="DIV2K_defscan")
    parser.add_argument("--project_path", default=os.getcwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--snr", type=int, default=10)
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--full_channel", action="store_true")
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    sys.path.insert(0, project_path)
    os.chdir(project_path)

    config = check_config(project_path, args.config_name)
    check_defmamba_branch(args.device)
    if not args.skip_model:
        check_model_forward(
            config,
            args.device,
            args.image_size,
            args.snr,
            args.full_channel,
        )


if __name__ == "__main__":
    main()
