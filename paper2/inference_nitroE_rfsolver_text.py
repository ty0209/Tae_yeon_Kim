#!/usr/bin/env python
# coding=utf-8
"""Text-prompt RF-Solver inference for vanilla Nitro-E (Llama 3.2 text encoder).

This is the text-prompt counterpart of inference_nitroE_rfsolver.py:
  - inference_nitroE_rfsolver.py        : EditCLIP exemplar-pair conditioning
                                          (uses fine-tuned EditCLIP + adapter)
  - inference_nitroE_rfsolver_text.py   : Llama-3.2-1B prompt conditioning
                                          (uses the base Nitro-E model only)

The RF-Solver math is identical to inference_nitroE_rfsolver.py — a 2nd-order
Taylor ODE step that uses one extra probe forward pass to estimate v'(z, t)
via finite difference (Wang et al., ICML 2025, Eq. 8/9). What changes is that
conditioning comes from Llama prompts (with attention_mask) instead of
EditCLIP image features.

Modes
-----
  recon  : null-cond inversion then null-cond reverse. Measures how
           reversibility-faithful the base Nitro-E + RF-Solver pipeline is on
           your image. No editing happens.

  edit   : source-prompt inversion, then target-prompt reverse with CFG.
           This is the standard text-prompt RF-Solver editing recipe.

Compared to inference_nitroE_rfinversion.py (text RF-Inversion via Algo 2),
RF-Solver uses higher-order ODE integration on both directions and does NOT
need an `eta` controller — preservation comes from the much more accurate
inversion-then-reverse trajectory of the 2nd-order solver.

Usage
-----
  # Reconstruction sanity check (no editing)
  python inference_nitroE_rfsolver_text.py \
      --input  /path/to/image.jpg \
      --mode   recon \
      --steps_list 30 \
      --output_dir output/rfsolver_text_recon

  # Text-prompt editing
  python inference_nitroE_rfsolver_text.py \
      --input          /path/to/image.jpg \
      --mode           edit \
      --source_prompt  "a photo of a brown pokemon figure" \
      --target_prompt  "a photo of a pink pokemon figure" \
      --steps_list     30 \
      --cfg_scale_edit 4.5 \
      --output_dir     output/rfsolver_text_edit
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
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.transformer_emmdit import EMMDiTTransformer


# =============================================================================
# Model loading (matches inference_nitroE_rfinversion.py)
# =============================================================================

def load_models(repo_name: str, ckpt_name: str, resolution: int,
                device: torch.device, dtype: torch.dtype):
    print(f"[load] tokenizer + Llama 3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", torch_dtype=dtype
    ).to(device).eval()

    print(f"[load] DC-AE VAE (32x)")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=dtype
    ).to(device).eval()

    print(f"[load] Nitro-E transformer ({repo_name}/{ckpt_name})")
    use_sub_attn = resolution <= 512
    transformer = EMMDiTTransformer(
        sample_size=resolution // 32, use_sub_attn=use_sub_attn,
    )
    state_dict = load_file(hf_hub_download(repo_name, ckpt_name))
    transformer.load_state_dict(state_dict)
    transformer = transformer.to(device, dtype=dtype).eval()

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    return tokenizer, text_encoder, vae, transformer, scheduler


# =============================================================================
# Prompt encoding (Llama 3.2 last hidden state)
# =============================================================================

@torch.no_grad()
def encode_prompt(prompt: str, tokenizer, text_encoder,
                  device: torch.device, dtype: torch.dtype,
                  max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    text_inputs = tokenizer(
        [prompt.lower().strip()],
        padding="max_length", max_length=max_length,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    out = text_encoder(input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    return out["hidden_states"][-1].to(dtype=dtype), attention_mask


# =============================================================================
# Image <-> latent
# =============================================================================

def load_image(path: str, resolution: int) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS)
    return img, transforms.ToTensor()(img).unsqueeze(0)


@torch.no_grad()
def encode_to_latent(pixels_01, vae, device, dtype):
    pixels = (pixels_01 * 2.0 - 1.0).to(device, dtype=dtype)
    out = vae.encode(pixels)
    z = (out.latent_dist.sample() if hasattr(out, "latent_dist")
         else out.latent if hasattr(out, "latent")
         else out.sample if hasattr(out, "sample") else out)
    s = getattr(vae, "scaling_factor", None) or vae.config.scaling_factor
    return z * s


@torch.no_grad()
def decode_from_latent(latents, vae):
    s = getattr(vae, "scaling_factor", None) or vae.config.scaling_factor
    out = vae.decode(latents / s)
    if hasattr(out, "sample"):
        out = out.sample
    return (out.float().clamp(-1, 1) + 1.0) / 2.0


# =============================================================================
# RF-Solver core (identical to inference_nitroE_rfsolver.py, but propagates
# the encoder_attention_mask required by Llama-conditioned Nitro-E).
# =============================================================================
#
# 2nd-order Taylor update (paper Eq. 8 / 9):
#   z_{t_next} = z_t + dt * v(z_t, t)
#              + 0.5 * dt^2 * v'(z_t, t)
# v' approximated by:
#   z_probe   = z_t + delta * v(z_t, t)
#   v_probe   = v(z_probe, t + delta)
#   v'(z_t,t) ≈ (v_probe - v_cur) / delta
#
# Setting order=1 recovers vanilla Euler.
# =============================================================================

@torch.no_grad()
def predict_velocity_cfg(transformer, x, cond_embeds, cond_mask,
                          null_embeds, null_mask, t_sched, cfg_scale, do_cfg):
    """One transformer forward with optional CFG over the two text branches."""
    batch = x.shape[0]
    t_in = t_sched.expand(batch).to(x.device)
    if do_cfg and cfg_scale > 1.0:
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t_in, t_in], dim=0)
        c_cat = torch.cat([null_embeds, cond_embeds], dim=0)
        m_cat = torch.cat([null_mask, cond_mask], dim=0)
        v_all = transformer(
            hidden_states=x_cat,
            encoder_hidden_states=c_cat,
            encoder_attention_mask=m_cat,
            timestep=t_cat,
            return_dict=False,
        )
        if isinstance(v_all, tuple):
            v_all = v_all[0]
        v_un, v_co = v_all.chunk(2)
        return v_un + cfg_scale * (v_co - v_un)
    out = transformer(
        hidden_states=x,
        encoder_hidden_states=cond_embeds,
        encoder_attention_mask=cond_mask,
        timestep=t_in,
        return_dict=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.no_grad()
def rfsolver_step(transformer, x, cond_embeds, cond_mask, null_embeds, null_mask,
                  t_cur, t_next, delta_norm, cfg_scale, order, do_cfg):
    """One RF-Solver step from t_cur to t_next (scheduler units [0, 1000])."""
    t_cur_f, t_next_f = float(t_cur), float(t_next)
    dt_norm = (t_next_f - t_cur_f) / 1000.0          # signed normalized step

    t_cur_t = torch.tensor([t_cur_f], device=x.device)
    v_cur = predict_velocity_cfg(
        transformer, x, cond_embeds, cond_mask,
        null_embeds, null_mask, t_cur_t, cfg_scale, do_cfg,
    )
    if order == 1:
        return x + dt_norm * v_cur

    # 2nd-order: estimate v' by probing one delta ahead in the same direction.
    sign = 1.0 if dt_norm >= 0 else -1.0
    delta_signed = delta_norm * sign
    delta_in_sched = delta_signed * 1000.0

    x_probe = x + delta_signed * v_cur
    t_probe_f = max(0.0, min(999.0, t_cur_f + delta_in_sched))
    t_probe_t = torch.tensor([t_probe_f], device=x.device)
    v_probe = predict_velocity_cfg(
        transformer, x_probe, cond_embeds, cond_mask,
        null_embeds, null_mask, t_probe_t, cfg_scale, do_cfg,
    )
    v_deriv = (v_probe - v_cur) / delta_signed
    return x + dt_norm * v_cur + 0.5 * (dt_norm ** 2) * v_deriv


@torch.no_grad()
def rfsolver_inversion(transformer, scheduler, x0,
                       cond_embeds, cond_mask, null_embeds, null_mask,
                       num_steps, delta_norm, order, cfg_scale, device):
    """clean -> noise. Typically uses null_embeds (or source-prompt) as cond,
    cfg_scale=1.0 (no CFG during inversion is the standard recipe).
    """
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_asc = scheduler.timesteps.flip(0).tolist() + [1000.0]
    x = x0.clone()
    do_cfg = cfg_scale > 1.0
    for i in tqdm(range(len(timesteps_asc) - 1),
                  desc="RF-Solver inversion", leave=False):
        x = rfsolver_step(
            transformer, x, cond_embeds, cond_mask, null_embeds, null_mask,
            t_cur=timesteps_asc[i], t_next=timesteps_asc[i + 1],
            delta_norm=delta_norm, cfg_scale=cfg_scale, order=order,
            do_cfg=do_cfg,
        )
    return x


@torch.no_grad()
def rfsolver_denoising(transformer, scheduler, xT,
                       cond_embeds, cond_mask, null_embeds, null_mask,
                       num_steps, delta_norm, order, cfg_scale, device):
    """noise -> clean. Uses target/source prompt + CFG."""
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_desc = scheduler.timesteps.tolist() + [0.0]
    x = xT.clone()
    do_cfg = cfg_scale > 1.0
    for i in tqdm(range(len(timesteps_desc) - 1),
                  desc="RF-Solver denoising", leave=False):
        x = rfsolver_step(
            transformer, x, cond_embeds, cond_mask, null_embeds, null_mask,
            t_cur=timesteps_desc[i], t_next=timesteps_desc[i + 1],
            delta_norm=delta_norm, cfg_scale=cfg_scale, order=order,
            do_cfg=do_cfg,
        )
    return x


# =============================================================================
# Helpers
# =============================================================================

def to_pil(t):
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


def psnr(a, b):
    mse = F.mse_loss(a.float().clamp(0, 1), b.float().clamp(0, 1)).item()
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def _resolve_inputs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sorted([q for q in p.iterdir() if q.suffix.lower() in exts])
    raise FileNotFoundError(path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True,
                        help="Single image path OR a directory of images.")
    parser.add_argument("--output_dir", type=str,
                        default="./output/rfsolver_text_test")
    parser.add_argument("--mode", type=str, choices=["recon", "edit"],
                        default="recon")

    # Model
    parser.add_argument("--repo_name", type=str, default="amd/Nitro-E")
    parser.add_argument("--ckpt_name", type=str,
                        default="Nitro-E-512px.safetensors")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    # RF-Solver
    parser.add_argument("--solver_order", type=int, choices=[1, 2], default=2,
                        help="1 = vanilla Euler RF, 2 = RF-Solver 2nd-order Taylor.")
    parser.add_argument("--delta_t", type=float, default=0.01,
                        help="Normalized probe size for v' finite difference. "
                             "Only used when solver_order=2.")
    parser.add_argument("--steps_list", type=int, nargs="+", default=[30],
                        help="Number of inversion+denoising steps. Several values "
                             "can be passed to sweep.")

    # Conditioning
    parser.add_argument("--cond_mode", type=str, default="empty",
                        choices=["empty", "zero"],
                        help="What to use for the null/uncond branch. 'empty' "
                             "encodes the literal empty string through Llama; "
                             "'zero' uses zeros with the same shape.")

    # Editing mode
    parser.add_argument("--source_prompt", type=str, default="",
                        help="Describes the original image. Used in --mode edit "
                             "as conditioning during inversion.")
    parser.add_argument("--target_prompt", type=str, default="",
                        help="Describes the edit target. Used in --mode edit "
                             "as conditioning during denoising.")
    parser.add_argument("--cfg_scale_invert", type=float, default=1.0,
                        help="CFG during inversion. Standard recipe uses 1.0 "
                             "(no CFG). Higher pushes inversion harder along the "
                             "source-prompt direction; risky.")
    parser.add_argument("--cfg_scale_edit", type=float, default=4.5,
                        help="CFG during denoising in edit mode.")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    tokenizer, text_encoder, vae, transformer, scheduler = load_models(
        args.repo_name, args.ckpt_name, args.resolution, device, dtype,
    )

    # null branch (= "empty" by default, or all-zero)
    empty_e, empty_m = encode_prompt("", tokenizer, text_encoder, device, dtype)
    if args.cond_mode == "zero":
        null_embeds = torch.zeros_like(empty_e)
        null_mask = torch.ones_like(empty_m)
    else:
        null_embeds, null_mask = empty_e, empty_m
    print(f"[cond] null branch '{args.cond_mode}': "
          f"shape={tuple(null_embeds.shape)}")

    # editing prompts
    if args.mode == "edit":
        if not args.target_prompt:
            raise SystemExit("--mode edit requires --target_prompt.")
        src_embeds, src_mask = encode_prompt(
            args.source_prompt, tokenizer, text_encoder, device, dtype)
        tgt_embeds, tgt_mask = encode_prompt(
            args.target_prompt, tokenizer, text_encoder, device, dtype)
        print(f"[cond] source: {args.source_prompt!r}")
        print(f"[cond] target: {args.target_prompt!r}")
    else:
        src_embeds, src_mask = None, None
        tgt_embeds, tgt_mask = None, None

    # -------------------------------------------------------------------------
    # Per-image loop
    # -------------------------------------------------------------------------
    image_paths = _resolve_inputs(args.input)
    print(f"[input] {len(image_paths)} image(s)")
    all_results = []

    for img_path in image_paths:
        name = img_path.stem
        print(f"\n=== {name} ===")
        orig_pil, orig_tensor = load_image(str(img_path), args.resolution)
        z0 = encode_to_latent(orig_tensor, vae, device, dtype)
        vae_recon = decode_from_latent(z0, vae).cpu()
        vae_p = psnr(orig_tensor, vae_recon)
        print(f"[ref] VAE-only PSNR={vae_p:.2f}")

        for n_steps in args.steps_list:
            # ----------------- inversion -----------------
            inv_cond = src_embeds if args.mode == "edit" else null_embeds
            inv_mask = src_mask   if args.mode == "edit" else null_mask
            print(f"\n[invert] n={n_steps}  order={args.solver_order}  "
                  f"delta_t={args.delta_t}  inv_cond={'source' if args.mode=='edit' else args.cond_mode}  "
                  f"cfg_invert={args.cfg_scale_invert}")
            xT = rfsolver_inversion(
                transformer, scheduler, z0,
                cond_embeds=inv_cond, cond_mask=inv_mask,
                null_embeds=null_embeds, null_mask=null_mask,
                num_steps=n_steps, delta_norm=args.delta_t,
                order=args.solver_order, cfg_scale=args.cfg_scale_invert,
                device=device,
            )
            print(f"[invert] xT mean={xT.float().mean().item():.3f} "
                  f"std={xT.float().std().item():.3f}   (target ~ N(0,1))")

            # ----------------- reconstruction reverse (always) -----------------
            # In edit mode, use the SAME conditioning as inversion (the source
            # prompt) so the recon trajectory stays in the same basin as
            # inversion. Reversing a source-cond-inverted xT under NULL would
            # drift to Nitro-E's unconditional manifold (often a blue sky /
            # empty scene), making the recon panel meaningless.
            if args.mode == "edit":
                recon_cond, recon_mask = src_embeds, src_mask
                recon_label = "source-cond"
            else:
                recon_cond, recon_mask = null_embeds, null_mask
                recon_label = "null-cond"
            print(f"[recon] {recon_label} denoise from xT")
            z_recon = rfsolver_denoising(
                transformer, scheduler, xT,
                cond_embeds=recon_cond, cond_mask=recon_mask,
                null_embeds=null_embeds, null_mask=null_mask,
                num_steps=n_steps, delta_norm=args.delta_t,
                order=args.solver_order, cfg_scale=1.0, device=device,
            )
            recon_pixels = decode_from_latent(z_recon, vae).cpu()
            recon_psnr = psnr(orig_tensor, recon_pixels)
            latent_mse = F.mse_loss(z0.float(), z_recon.float()).item()
            print(f"[recon] PSNR={recon_psnr:.2f}  latentMSE={latent_mse:.5f}  "
                  f"(VAE ceiling: {vae_p:.2f})")

            out_pils  = [orig_pil, to_pil(vae_recon), to_pil(recon_pixels)]
            out_labels = ["orig", "VAE-only", f"recon(n={n_steps})\nPSNR={recon_psnr:.1f}"]

            # ----------------- editing reverse (only in edit mode) -----------------
            edit_psnr = None
            if args.mode == "edit":
                print(f"[edit]  target-cond denoise from xT, cfg={args.cfg_scale_edit}")
                z_edit = rfsolver_denoising(
                    transformer, scheduler, xT,
                    cond_embeds=tgt_embeds, cond_mask=tgt_mask,
                    null_embeds=null_embeds, null_mask=null_mask,
                    num_steps=n_steps, delta_norm=args.delta_t,
                    order=args.solver_order, cfg_scale=args.cfg_scale_edit,
                    device=device,
                )
                edit_pixels = decode_from_latent(z_edit, vae).cpu()
                edit_psnr = psnr(orig_tensor, edit_pixels)
                print(f"[edit]  PSNR vs original={edit_psnr:.2f} "
                      f"(lower = more edited)")
                out_pils.append(to_pil(edit_pixels))
                out_labels.append(f"edit(n={n_steps})\ncfg={args.cfg_scale_edit}")

                edit_pixels_pil = to_pil(edit_pixels)
                edit_save = (f"{name}_edit_n{n_steps}_order{args.solver_order}"
                             f"_cfg{args.cfg_scale_edit}.png")
                edit_pixels_pil.save(os.path.join(args.output_dir, edit_save))

            # save grid
            tag = f"{name}_n{n_steps}_order{args.solver_order}"
            if args.mode == "edit":
                tag += f"_cfgE{args.cfg_scale_edit}"
            grid = make_grid(out_pils, out_labels)
            grid.save(os.path.join(args.output_dir, f"{tag}_grid.png"))

            # save recon alone
            to_pil(recon_pixels).save(
                os.path.join(args.output_dir,
                              f"{name}_recon_n{n_steps}_order{args.solver_order}.png"))

            all_results.append({
                "name": name,
                "n_steps": n_steps,
                "order": args.solver_order,
                "vae_psnr": vae_p,
                "recon_psnr": recon_psnr,
                "latent_mse": latent_mse,
                "edit_psnr_vs_original": edit_psnr,
            })

    # ---------------- summary ----------------
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\n[done] -> {args.output_dir}")
    print("       grid layout: [orig | VAE-only | recon |"
          + (" edit |" if args.mode == "edit" else "")
          + "]")


if __name__ == "__main__":
    main()
