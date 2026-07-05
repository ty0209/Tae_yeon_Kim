#!/usr/bin/env python
# coding=utf-8
"""
Inference script for fine-tuned EditCLIP + E-MMDiT (Nitro-E) using SDEdit.

Flow
----
1. Load fine-tuned Nitro-E transformer (checkpoint-XXXX/transformer.pt).
2. Load the EditCLIP -> Nitro-E adapter (checkpoint-XXXX/editclip_adapter.pt).
3. Encode exemplar pair (before, after) via EditCLIP (6-ch concat) -> adapter
   -> editing embedding `edit_embeds` of shape [1, 256, 2048].
4. Encode query image via DC-AE -> latent x_0.
5. SDEdit: x_sigma = (1-sigma)*x_0 + sigma*eps, then reverse ODE conditioned on
   edit_embeds with CFG using zero null embeds for the uncond branch.
6. Decode and save.

Why SDEdit instead of RF-Inversion?
-----------------------------------
ODE inversion through Nitro-E's ASA + multi-path token compression has
reconstruction issues (PSNR ~17 dB in our sanity check). SDEdit skips inversion
entirely — partial noise + denoise from a chosen sigma — so reversibility of
the architecture is not required. Structure preservation comes from `strength`
(low strength = keep more of the original, high strength = more aggressive edit).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
from transformers import AutoProcessor, CLIPModel

from core.models.transformer_emmdit import EMMDiTTransformer


# =============================================================================
# Model loading
# =============================================================================

def load_models(
    finetuned_ckpt: str,
    editclip_path: str,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Returns transformer, editclip_adapter, vae, editclip_encoder, editclip_processor, scheduler."""
    print(f"[load] E-MMDiT transformer (fine-tuned: {finetuned_ckpt})")
    transformer = EMMDiTTransformer(
        sample_size=resolution // 32,
        use_sub_attn=(resolution <= 512),
    )
    state = torch.load(finetuned_ckpt, map_location="cpu")
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    # adapter sits next to transformer.pt
    adapter_path = os.path.join(os.path.dirname(finetuned_ckpt), "editclip_adapter.pt")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"editclip_adapter.pt not found at {adapter_path}. "
            f"Make sure the training step saved it next to transformer.pt."
        )
    adapter_state = torch.load(adapter_path, map_location="cpu")
    # Sequential keys: '0.weight','0.bias' (Linear), '1.weight','1.bias' (LayerNorm)
    editclip_dim = adapter_state["0.weight"].shape[1]
    caption_dim = adapter_state["0.weight"].shape[0]
    print(f"[load] EditCLIP adapter: Linear({editclip_dim} -> {caption_dim}) + LN")
    editclip_adapter = nn.Sequential(
        nn.Linear(editclip_dim, caption_dim),
        nn.LayerNorm(caption_dim),
    )
    editclip_adapter.load_state_dict(adapter_state)
    editclip_adapter = editclip_adapter.to(device, dtype=dtype).eval()

    print(f"[load] DC-AE VAE")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=dtype
    ).to(device).eval()

    print(f"[load] EditCLIP encoder ({editclip_path})")
    editclip_full = CLIPModel.from_pretrained(editclip_path)
    editclip_encoder = editclip_full.vision_model.to(device, dtype=dtype).eval()
    editclip_processor = AutoProcessor.from_pretrained(editclip_path)
    del editclip_full

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    return transformer, editclip_adapter, vae, editclip_encoder, editclip_processor, scheduler


# =============================================================================
# Image / latent helpers
# =============================================================================

def load_image(path: str, resolution: int) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0)   # [1, 3, H, W] in [0, 1]
    return img, tensor


@torch.no_grad()
def encode_to_latent(pixels_01: torch.Tensor, vae, device, dtype) -> torch.Tensor:
    pixels = (pixels_01 * 2.0 - 1.0).to(device, dtype=dtype)   # [-1, 1]
    out = vae.encode(pixels)
    z = (out.latent_dist.sample() if hasattr(out, "latent_dist")
         else out.latent if hasattr(out, "latent")
         else out.sample if hasattr(out, "sample")
         else out)
    s = getattr(vae, "scaling_factor", None) or vae.config.scaling_factor
    return z * s


@torch.no_grad()
def decode_from_latent(latents: torch.Tensor, vae) -> torch.Tensor:
    s = getattr(vae, "scaling_factor", None) or vae.config.scaling_factor
    out = vae.decode(latents / s)
    if hasattr(out, "sample"):
        out = out.sample
    return (out.float().clamp(-1, 1) + 1.0) / 2.0


@torch.no_grad()
def compute_edit_embeds(
    exemplar_in_pil: Image.Image,
    exemplar_ed_pil: Image.Image,
    editclip_encoder,
    editclip_processor,
    editclip_adapter,
    device,
    dtype,
) -> torch.Tensor:
    """6-channel concat -> EditCLIP -> drop CLS -> adapter -> [1, 256, 2048]."""
    orig_clip = editclip_processor(images=exemplar_in_pil, return_tensors="pt")
    edit_clip = editclip_processor(images=exemplar_ed_pil, return_tensors="pt")
    concat_clip = torch.cat(
        [orig_clip.pixel_values, edit_clip.pixel_values], dim=1
    ).to(device, dtype=dtype)
    # last_hidden_state: [1, 257, 1024] for ViT-L/14 (1 CLS + 256 patches)
    feats = editclip_encoder(concat_clip)[0][:, 1:, :]   # drop CLS -> [1, 256, 1024]
    embeds = editclip_adapter(feats)                      # [1, 256, 2048]
    return embeds


# =============================================================================
# SDEdit core (Rectified Flow)
# =============================================================================

@torch.no_grad()
def sdedit_reverse(
    transformer,
    scheduler,
    x0_latents: torch.Tensor,
    cond_embeds: torch.Tensor,
    null_embeds: torch.Tensor,
    strength: float,
    num_inference_steps: int,
    cfg_scale: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """strength in [0, 1]: 0 = no edit, 1 = full text-to-image generation."""
    assert 0.0 <= strength <= 1.0

    scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = scheduler.timesteps   # descending

    init_idx = min(int(round(num_inference_steps * (1.0 - strength))), num_inference_steps - 1)
    init_idx = max(init_idx, 0)
    timesteps = all_timesteps[init_idx:]
    if len(timesteps) == 0:
        return x0_latents.clone()

    # Add noise at the starting sigma.
    sigma_start = timesteps[0].float() / 1000.0
    if generator is not None:
        eps = torch.randn(
            x0_latents.shape, generator=generator,
            device=x0_latents.device, dtype=x0_latents.dtype,
        )
    else:
        eps = torch.randn_like(x0_latents)
    x = (1.0 - sigma_start) * x0_latents + sigma_start * eps

    batch = x.shape[0]
    do_cfg = cfg_scale > 1.0

    for i, t in enumerate(tqdm(timesteps, desc=f"SDEdit (s={strength:.2f})", leave=False)):
        t_in = t.float().unsqueeze(0).expand(batch).to(device)

        if do_cfg:
            x_cat = torch.cat([x, x], dim=0)
            t_cat = torch.cat([t_in, t_in], dim=0)
            c_cat = torch.cat([null_embeds, cond_embeds], dim=0)
            v_all = transformer(
                hidden_states=x_cat,
                encoder_hidden_states=c_cat,
                timestep=t_cat,
                return_dict=False,
            )
            if isinstance(v_all, tuple):
                v_all = v_all[0]
            v_uncond, v_cond = v_all.chunk(2)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = transformer(
                hidden_states=x,
                encoder_hidden_states=cond_embeds,
                timestep=t_in,
                return_dict=False,
            )
            if isinstance(v, tuple):
                v = v[0]

        if i < len(timesteps) - 1:
            dt = (timesteps[i + 1] - t).float() / 1000.0   # negative
        else:
            dt = -t.float() / 1000.0
        x = x + dt * v

    return x


# =============================================================================
# Visualization
# =============================================================================

def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_grid(images: List[Image.Image], labels: List[str]) -> Image.Image:
    assert len(images) == len(labels)
    w, h = images[0].size
    pad = 24
    grid = Image.new("RGB", (w * len(images), h + pad), color=(255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img, (i * w, pad))
    draw = ImageDraw.Draw(grid)
    for i, lab in enumerate(labels):
        draw.text((i * w + 4, 4), lab, fill=(0, 0, 0))
    return grid


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # weights
    parser.add_argument("--finetuned_ckpt", type=str, required=True,
                        help="Path to fine-tuned transformer.pt. "
                             "editclip_adapter.pt is auto-located next to it.")
    parser.add_argument("--editclip_path", type=str,
                        default="/root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14",
                        help="HF-format EditCLIP folder (with config.json + model.safetensors).")
    # inputs
    parser.add_argument("--exemplar_input", type=str, required=True,
                        help="Path to exemplar BEFORE-edit image.")
    parser.add_argument("--exemplar_edited", type=str, required=True,
                        help="Path to exemplar AFTER-edit image.")
    parser.add_argument("--query_image", type=str, required=True,
                        help="Path to query image to edit.")
    parser.add_argument("--output_dir", type=str, default="output/inference_sdedit")
    # SDEdit knobs
    parser.add_argument("--strength_list", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="SDEdit strengths to sweep. Higher = more edit, less structure.")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    # misc
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    transformer, editclip_adapter, vae, editclip_encoder, editclip_processor, scheduler = load_models(
        args.finetuned_ckpt, args.editclip_path, args.resolution, device, dtype,
    )

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------
    print(f"[input] exemplar_input : {args.exemplar_input}")
    print(f"[input] exemplar_edited: {args.exemplar_edited}")
    print(f"[input] query_image    : {args.query_image}")
    exemplar_in_pil, _ = load_image(args.exemplar_input, args.resolution)
    exemplar_ed_pil, _ = load_image(args.exemplar_edited, args.resolution)
    query_pil, query_tensor = load_image(args.query_image, args.resolution)

    # -------------------------------------------------------------------------
    # Conditioning embeddings
    # -------------------------------------------------------------------------
    print("[cond] computing EditCLIP -> adapter embeddings...")
    edit_embeds = compute_edit_embeds(
        exemplar_in_pil, exemplar_ed_pil,
        editclip_encoder, editclip_processor, editclip_adapter,
        device, dtype,
    )
    null_embeds = torch.zeros_like(edit_embeds)
    print(f"[cond] edit_embeds shape: {tuple(edit_embeds.shape)}")

    # -------------------------------------------------------------------------
    # Query -> latent
    # -------------------------------------------------------------------------
    x0 = encode_to_latent(query_tensor, vae, device, dtype)
    # VAE-only baseline for reference (how lossy is the VAE at this image)
    vae_recon = decode_from_latent(x0, vae).cpu()

    # -------------------------------------------------------------------------
    # SDEdit sweep
    # -------------------------------------------------------------------------
    out_pils = [exemplar_in_pil, exemplar_ed_pil, query_pil, to_pil(vae_recon)]
    out_labels = ["ex-in", "ex-ed", "query", "VAE-only"]
    results = []

    for strength in args.strength_list:
        print(f"\n[SDEdit] strength={strength}, steps={args.num_inference_steps}, "
              f"cfg={args.cfg_scale}")
        edited_latents = sdedit_reverse(
            transformer, scheduler, x0,
            edit_embeds, null_embeds,
            strength=strength,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            device=device,
            generator=gen,
        )
        edited_pixels = decode_from_latent(edited_latents, vae).cpu()
        edited_pil = to_pil(edited_pixels)

        out_pils.append(edited_pil)
        out_labels.append(f"s={strength}")
        edited_pil.save(os.path.join(args.output_dir, f"edited_s{strength}.png"))
        results.append({
            "strength": strength,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
        })

    # -------------------------------------------------------------------------
    # Save grid + summary
    # -------------------------------------------------------------------------
    grid = make_grid(out_pils, out_labels)
    grid.save(os.path.join(args.output_dir, "grid.png"))

    summary = {
        "args": vars(args),
        "edit_embeds_shape": list(edit_embeds.shape),
        "results": results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.output_dir}")
    print("       grid layout: [ex-in | ex-ed | query | VAE-only | s=... | s=... | ...]")


if __name__ == "__main__":
    main()
