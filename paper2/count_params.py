#!/usr/bin/env python
# coding=utf-8
"""
Count parameter sizes for the EditCLIP + InstructPix2Pix (paper) baseline.

Pipeline modules counted:
  1. SD-1.5 IP2P U-Net               (from editclip_ip2p/unet/)
  2. SD-1.5 VAE                      (from editclip_ip2p/vae/)
  3. CLIP text encoder (IP2P legacy) (from editclip_ip2p/text_encoder/)
  4. ImageProjModel (1024→768 + LN)  (from editclip_ip2p/image_proj_model.pt)
  5. EditCLIP vision encoder         (from editclip_vit_l_14/)
     also reports visual_projection.

Frozen at inference (no training mode). Reports per-module total +
approximate fp16 memory.

Usage
-----
python count_params_paper.py \
    --ip2p_ckpt_dir /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_ip2p \
    --editclip_path /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDITCLIP_SRC = "/root/e_mmdit/flash-attention/EditCLIP/src/sd_ip2p_train"
if _EDITCLIP_SRC not in sys.path:
    sys.path.insert(0, _EDITCLIP_SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn


def count_module_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def fmt_count(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} B"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def fmt_bytes(n_bytes: float) -> str:
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / (1024 ** 3):.3f} GiB"
    if n_bytes >= 1024 ** 2:
        return f"{n_bytes / (1024 ** 2):.3f} MiB"
    return f"{n_bytes / 1024:.3f} KiB"


def _print_row(name: str, total: int, trainable: int = 0,
                indent: int = 0, dtype_bytes: int = 2):
    pad = "  " * indent
    print(f"  {pad}{name:<45} "
          f"total={fmt_count(total):>14}   "
          f"trainable={fmt_count(trainable):>10}   "
          f"~mem(bf16/fp16)={fmt_bytes(total * dtype_bytes):>12}")


def report_unet(ip2p_ckpt_dir: str) -> int:
    from diffusers import UNet2DConditionModel
    path = os.path.join(ip2p_ckpt_dir, "unet")
    print(f"\n[paper / IP2P] SD-1.5 U-Net (modified for IP2P channel concat)")
    print(f"  path: {path}")
    unet = UNet2DConditionModel.from_pretrained(path)
    total, _ = count_module_params(unet)
    _print_row("U-Net (whole)", total)
    # Quick structural breakdown.
    sub = {
        "conv_in (in_channels={})".format(unet.config.in_channels):
            unet.conv_in,
        "time_embedding": unet.time_embedding,
        "down_blocks": unet.down_blocks,
        "mid_block":   unet.mid_block,
        "up_blocks":   unet.up_blocks,
        "conv_out":    unet.conv_out,
    }
    accounted = 0
    for name, m in sub.items():
        n, _ = count_module_params(m)
        _print_row(name, n, indent=1)
        accounted += n
    if total > accounted:
        _print_row("[unaccounted: norm/conv_norm_out/etc.]",
                     total - accounted, indent=1)
    del unet
    return total


def report_vae(ip2p_ckpt_dir: str) -> int:
    from diffusers import AutoencoderKL
    path = os.path.join(ip2p_ckpt_dir, "vae")
    print(f"\n[paper / IP2P] SD-1.5 VAE (AutoencoderKL)")
    print(f"  path: {path}")
    vae = AutoencoderKL.from_pretrained(path)
    total, _ = count_module_params(vae)
    _print_row("VAE (whole)", total)
    del vae
    return total


def report_text_encoder(ip2p_ckpt_dir: str) -> int:
    """CLIP text encoder from the underlying SD pipeline. Note: the paper's
    EditCLIP-transfer pipeline FEEDS prompt_embeds directly (from EditCLIP
    vision via ImageProjModel), so this text encoder is unused at
    inference. Counted for completeness.
    """
    from transformers import CLIPTextModel
    path = os.path.join(ip2p_ckpt_dir, "text_encoder")
    print(f"\n[paper / IP2P] CLIP text encoder (UNUSED at inference)")
    print(f"  path: {path}")
    te = CLIPTextModel.from_pretrained(path)
    total, _ = count_module_params(te)
    _print_row("CLIP text encoder (whole)", total)
    del te
    return total


def report_image_proj_model(ip2p_ckpt_dir: str,
                              editclip_path: str) -> int:
    """ImageProjModel: Linear(1024, 1*768) + LayerNorm(768).
    Implementation in src/sd_ip2p_train/projection_model.py.
    """
    from projection_model import ImageProjModel
    from diffusers import UNet2DConditionModel
    from transformers import CLIPModel

    # Need cross_attention_dim from unet, clip_embeddings_dim from EditCLIP.
    unet_cfg_path = os.path.join(ip2p_ckpt_dir, "unet")
    unet = UNet2DConditionModel.from_pretrained(unet_cfg_path)
    cross_dim = unet.config.cross_attention_dim
    del unet

    editclip = CLIPModel.from_pretrained(editclip_path)
    clip_dim = editclip.config.vision_config.hidden_size
    del editclip

    proj = ImageProjModel(
        cross_attention_dim=cross_dim,
        clip_embeddings_dim=clip_dim,
        clip_extra_context_tokens=1,
    )
    state_path = os.path.join(ip2p_ckpt_dir, "image_proj_model.pt")
    if os.path.exists(state_path):
        proj.load_state_dict(torch.load(state_path, map_location="cpu"))
    total, _ = count_module_params(proj)
    print(f"\n[paper] ImageProjModel (Linear + LayerNorm)")
    print(f"  shape: Linear({clip_dim} -> {cross_dim}) + LN({cross_dim})")
    print(f"  state: {state_path}{' (loaded)' if os.path.exists(state_path) else ' (state missing, structure only)'}")
    _print_row("ImageProjModel (whole)", total)
    del proj
    return total


def report_editclip(editclip_path: str) -> int:
    from transformers import CLIPModel
    print(f"\n[paper] EditCLIP vision encoder (modified ViT-L/14, 6-ch input)")
    print(f"  path: {editclip_path}")
    full = CLIPModel.from_pretrained(editclip_path)
    vision_total, _ = count_module_params(full.vision_model)
    proj_total, _   = count_module_params(full.visual_projection)
    _print_row("vision_model (used at inference)", vision_total)
    _print_row("visual_projection (used at inference for cond)",
                 proj_total, indent=1)
    del full
    return vision_total + proj_total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip2p_ckpt_dir", type=str, required=True,
                   help="editclip_ip2p/ — must contain unet/, vae/, "
                        "text_encoder/, image_proj_model.pt.")
    p.add_argument("--editclip_path", type=str, required=True,
                   help="EditCLIP visual encoder HF dir.")
    p.add_argument("--include_text_encoder", action="store_true",
                   help="Count the CLIP text encoder too (unused at inference "
                        "for EditCLIP-transfer pipeline). Default OFF.")
    args = p.parse_args()

    print(f"\n=== EditCLIP + InstructPix2Pix (paper baseline) ===")
    print(f"  IP2P ckpt: {args.ip2p_ckpt_dir}")
    print(f"  EditCLIP : {args.editclip_path}")

    unet_total = report_unet(args.ip2p_ckpt_dir)
    vae_total  = report_vae(args.ip2p_ckpt_dir)
    text_total = 0
    if args.include_text_encoder:
        text_total = report_text_encoder(args.ip2p_ckpt_dir)
    proj_total = report_image_proj_model(args.ip2p_ckpt_dir,
                                            args.editclip_path)
    editclip_total = report_editclip(args.editclip_path)

    print(f"\n=== TOTAL (inference-active modules) ===")
    print(f"  EditCLIP encoder (vision + visual_projection) = "
          f"{fmt_count(editclip_total)}")
    print(f"  ImageProjModel                                = "
          f"{fmt_count(proj_total)}")
    print(f"  IP2P U-Net                                    = "
          f"{fmt_count(unet_total)}")
    print(f"  SD-1.5 VAE                                    = "
          f"{fmt_count(vae_total)}")
    if args.include_text_encoder:
        print(f"  CLIP text encoder (UNUSED)                    = "
              f"{fmt_count(text_total)}")
    s = editclip_total + proj_total + unet_total + vae_total
    print(f"  ----------")
    print(f"  TOTAL (active)                                = "
          f"{fmt_count(s)}  (~{fmt_bytes(s * 2)} bf16)")
    if args.include_text_encoder:
        s2 = s + text_total
        print(f"  TOTAL (with text encoder included)            = "
              f"{fmt_count(s2)}  (~{fmt_bytes(s2 * 2)} bf16)")
    print()


if __name__ == "__main__":
    main()
