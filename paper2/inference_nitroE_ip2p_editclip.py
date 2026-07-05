#!/usr/bin/env python
# coding=utf-8
"""
Inference for **Stage 2 IP2P+EditCLIP** checkpoints — Nitro-E with
channel concat + EditCLIP-adapter conditioning.

What this script does
---------------------
The Stage 2 trainer (train_editclip_emmdit_ip2p_stage2.py) produces a
checkpoint with:
  * 64-channel pos_embed.proj (IP2P-style channel concat extension)
  * Transformer trained to read [x_t || x0_source] on its hidden input
  * EditCLIP visual encoder (frozen) + fresh adapter (Linear/MLP) for cond
  * Text-stream adapted to the adapter's output distribution

At inference we replicate that input format:

    hidden_in = concat([x_t, x0_source_clean], dim=channel)   # [B, 64, H, W]
    cond_emb  = adapter(EditCLIP(exemplar_input, exemplar_edited))
    v_pred    = transformer(hidden_in, cond_emb, t)

Inversion is OPTIONAL here. IP2P style is to start from random noise and
let the channel concat carry source info; no RF-Inversion needed. We
also provide `--use_inversion` for deterministic mode (same source ->
same xT -> same output).

Pipeline
--------
1. Build Nitro-E base (32 ch) -> extend pos_embed.proj to 64 ch ->
   load Stage 2 transformer.pt (strict, expects 64 ch).
2. Load EditCLIP encoder (frozen) + editclip_adapter.pt from same ckpt dir.
3. Encode exemplar pair via EditCLIP+adapter -> edit_embeds.
4. VAE-encode query -> x0_source (clean, 32 ch).
5. xT = random noise (or inversion if --use_inversion).
6. RF-Solver denoise:
     for each step t -> t_next:
       hidden = concat([x_t, x0_source])     # IP2P concat
       v = transformer(hidden, cond, t)      # optional CFG inside
       x_t = x_t + dt * v
7. Decode -> edited image.

CFG modes
~~~~~~~~~
  * --cfg_mode text (default): single-CFG over text cond only. Source
    latent is ALWAYS present in the channel input (matches IP2P training
    when image_dropout misses).
  * --cfg_mode dual: IP2P paper's dual-CFG with separate text / image
    scales. Three forward passes per step (or batch x3). More accurate
    but 3x slower.

Offline mode
------------
All HF library calls are forced offline (HF_HUB_OFFLINE=1 etc.). The
script never reaches out to the network.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Force HF library calls offline BEFORE any HF import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

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
from transformers import AutoProcessor, CLIPModel

from core.models.transformer_emmdit import EMMDiTTransformer


# =============================================================================
# Cache-aware loaders (offline-only)
# =============================================================================

def _find_local_dcae() -> Optional[str]:
    repo = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    cache_root = Path(os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if not cache_dir.is_dir():
        return None
    snaps = cache_dir / "snapshots"
    if not snaps.is_dir():
        return None
    for sn in sorted(snaps.iterdir()):
        if (sn / "config.json").exists():
            return str(sn.resolve())
    return None


def _find_nitroE_weights(repo: str, filename: str,
                          local_path: Optional[str] = None) -> str:
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"--nitroE_local_path not found: {local_path}")
        return local_path
    cache_root = Path(os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if cache_dir.is_dir():
        snaps = cache_dir / "snapshots"
        if snaps.is_dir():
            for sn in sorted(snaps.iterdir()):
                cand = sn / filename
                if cand.is_file():
                    return str(cand.resolve())
    return hf_hub_download(repo, filename)


# =============================================================================
# Architecture extension — match Stage 2 trainer
# =============================================================================

def extend_pos_embed_to_concat(transformer) -> int:
    """32 -> 64 in_channels, zero-init for the new half. Idempotent."""
    old_proj = transformer.pos_embed.proj
    assert isinstance(old_proj, nn.Conv2d)
    old_in = old_proj.in_channels
    new_in = old_in * 2
    new_proj = nn.Conv2d(
        new_in, old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride, padding=old_proj.padding,
        bias=(old_proj.bias is not None),
    )
    with torch.no_grad():
        new_w = torch.zeros_like(new_proj.weight)
        new_w[:, :old_in] = old_proj.weight.data
        new_proj.weight.data.copy_(new_w)
        if old_proj.bias is not None:
            new_proj.bias.data.copy_(old_proj.bias.data)
    transformer.pos_embed.proj = new_proj
    transformer.config.in_channels = new_in
    transformer.register_to_config(in_channels=new_in)
    return new_in


# =============================================================================
# Adapter — auto-detect linear vs MLP from state_dict
# =============================================================================

def build_adapter_from_state(state):
    """Reconstruct the EditCLIP adapter from a saved state_dict.

    Conventions:
      MLP   (Stage 2 default):       '0' Linear, '1' GELU(no params),
                                     '2' Linear, '3' LayerNorm
      Linear+LN (Stage 2 alt):       '0' Linear, '1' LayerNorm
      Simple Linear (Stage 2 simple): '0' Linear only — no LN
                                     (saved by train_editclip_emmdit_ip2p_stage2_simple.py)
    """
    keys = set(state.keys())
    has_layer2 = "2.weight" in keys
    has_layer1 = "1.weight" in keys
    if has_layer2:
        editclip_dim = state["0.weight"].shape[1]
        hidden_dim = state["0.weight"].shape[0]
        caption_dim = state["2.weight"].shape[0]
        adapter = nn.Sequential(
            nn.Linear(editclip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, caption_dim),
            nn.LayerNorm(caption_dim),
        )
        print(f"[load] EditCLIP adapter [mlp]: "
              f"Linear({editclip_dim} -> {hidden_dim}) -> GELU -> "
              f"Linear({hidden_dim} -> {caption_dim}) + LN")
    elif has_layer1:
        editclip_dim = state["0.weight"].shape[1]
        caption_dim = state["0.weight"].shape[0]
        adapter = nn.Sequential(
            nn.Linear(editclip_dim, caption_dim),
            nn.LayerNorm(caption_dim),
        )
        print(f"[load] EditCLIP adapter [linear+ln]: "
              f"Linear({editclip_dim} -> {caption_dim}) + LN")
    else:
        editclip_dim = state["0.weight"].shape[1]
        caption_dim = state["0.weight"].shape[0]
        adapter = nn.Sequential(nn.Linear(editclip_dim, caption_dim))
        print(f"[load] EditCLIP adapter [simple linear]: "
              f"Linear({editclip_dim} -> {caption_dim})  (no LN)")
    adapter.load_state_dict(state)
    return adapter


# =============================================================================
# Model loading
# =============================================================================

def load_models(finetuned_ckpt: str, editclip_path: str,
                 resolution: int, device, dtype,
                 nitroE_local_path: Optional[str] = None,
                 dcae_local_path: Optional[str] = None):
    # ---- Transformer ----
    print(f"[load] base Nitro-E (32 ch) then extend to 64 ch")
    transformer = EMMDiTTransformer(
        sample_size=resolution // 32,
        use_sub_attn=(resolution <= 512),
    )
    nitroE_path = _find_nitroE_weights(
        "amd/Nitro-E", "Nitro-E-512px.safetensors",
        local_path=nitroE_local_path,
    )
    transformer.load_state_dict(load_file(nitroE_path))
    extend_pos_embed_to_concat(transformer)

    print(f"[load] Stage 2 transformer: {finetuned_ckpt}")
    state = torch.load(finetuned_ckpt, map_location="cpu")
    pe_in = state.get("pos_embed.proj.weight", None)
    if pe_in is None or pe_in.shape[1] != 64:
        raise RuntimeError(
            f"Stage 2 ckpt must have 64-ch pos_embed.proj.weight. "
            f"Got shape {None if pe_in is None else tuple(pe_in.shape)}. "
            f"Is this really a Stage-2 IP2P checkpoint?"
        )
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    # ---- Adapter (from same checkpoint dir) ----
    adapter_path = os.path.join(os.path.dirname(finetuned_ckpt),
                                  "editclip_adapter.pt")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"editclip_adapter.pt not found alongside "
                                   f"transformer.pt at {adapter_path}")
    adapter_state = torch.load(adapter_path, map_location="cpu")
    editclip_adapter = build_adapter_from_state(adapter_state)
    editclip_adapter = editclip_adapter.to(device, dtype=dtype).eval()

    # ---- DC-AE VAE (offline) ----
    if dcae_local_path is None:
        dcae_local_path = _find_local_dcae()
    dcae_src = (dcae_local_path
                or "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    print(f"[load] DC-AE VAE (local_files_only=True): {dcae_src}")
    vae = AutoencoderDC.from_pretrained(
        dcae_src, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()

    # ---- EditCLIP encoder + processor (frozen, offline) ----
    print(f"[load] EditCLIP encoder ({editclip_path})")
    editclip_full = CLIPModel.from_pretrained(
        editclip_path, local_files_only=True,
    )
    editclip_encoder = editclip_full.vision_model.to(
        device, dtype=dtype
    ).eval()
    editclip_processor = AutoProcessor.from_pretrained(
        editclip_path, local_files_only=True,
    )
    del editclip_full

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    return (transformer, editclip_adapter, vae,
            editclip_encoder, editclip_processor, scheduler)


# =============================================================================
# Image / latent / cond helpers
# =============================================================================

def load_image(path, resolution):
    img = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS
    )
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


@torch.no_grad()
def compute_edit_embeds(ex_in_pil, ex_ed_pil, encoder, processor, adapter,
                          device, dtype, adapter_scale: float = 1.0,
                          cond_token_mode: str = "patches"):
    """6-channel concat (exemplar pair) -> EditCLIP -> adapter.

    cond_token_mode:
      'patches' (default): drop CLS, use 256 patch tokens (16x16 spatial).
      'cls': take CLS only, replicate to 16 tokens (E-MMDiT ASA needs
              seq_len divisible by LCM(1, 4, 16) = 16). No spatial layout.
    """
    o = processor(images=ex_in_pil, return_tensors="pt")
    e = processor(images=ex_ed_pil, return_tensors="pt")
    concat = torch.cat([o.pixel_values, e.pixel_values], dim=1).to(
        device, dtype=dtype
    )
    raw = encoder(concat)[0]
    if cond_token_mode == "cls":
        cls = raw[:, 0:1, :]
        feats = cls.repeat(1, 16, 1)  # [B, 16, hidden]
    else:
        feats = raw[:, 1:, :]  # [B, 256, hidden]
    out = adapter(feats)
    if adapter_scale != 1.0:
        out = out * adapter_scale
    return out


# =============================================================================
# RF-Solver with channel concat
# =============================================================================

@torch.no_grad()
def _transformer_forward(transformer, hidden_in, encoder_hidden_states,
                          timestep):
    tdtype = next(transformer.parameters()).dtype
    out = transformer(
        hidden_states=hidden_in.to(tdtype),
        encoder_hidden_states=encoder_hidden_states.to(tdtype),
        timestep=timestep,
        return_dict=False,
    )
    return out[0] if isinstance(out, tuple) else out.sample


@torch.no_grad()
def predict_velocity_text_cfg(transformer, x_t, x0_source,
                                cond_embeds, null_embeds,
                                t_sched, cfg_scale):
    """Single-CFG over text only; source ALWAYS in channel input.

    Matches the dominant training regime: image_dropout misses ~95% of the
    time, so the model is most familiar with "source-present" inputs. CFG
    here amplifies the cond direction at a fixed structural anchor.
    """
    batch = x_t.shape[0]
    t_in = t_sched.expand(batch).to(x_t.device)
    hin = torch.cat([x_t, x0_source], dim=1)
    do_cfg = cfg_scale > 1.0
    if do_cfg:
        h_cat = torch.cat([hin, hin], dim=0)
        t_cat = torch.cat([t_in, t_in], dim=0)
        c_cat = torch.cat([null_embeds, cond_embeds], dim=0)
        v = _transformer_forward(transformer, h_cat, c_cat, t_cat)
        v_un, v_co = v.chunk(2)
        return v_un + cfg_scale * (v_co - v_un)
    return _transformer_forward(transformer, hin, cond_embeds, t_in)


@torch.no_grad()
def predict_velocity_dual_cfg(transformer, x_t, x0_source,
                                cond_embeds, null_embeds,
                                t_sched, cfg_text, cfg_image):
    """IP2P paper dual-CFG. 3 forward passes (or batch=3 stacked).

        v_uu = v( cat[x_t, 0], null_cond )   # text drop + image drop
        v_tu = v( cat[x_t, 0], cond )        # text only
        v_tt = v( cat[x_t, src], cond )      # both
        v = v_uu + s_T * (v_tu - v_uu) + s_I * (v_tt - v_tu)
    """
    batch = x_t.shape[0]
    t_in = t_sched.expand(batch).to(x_t.device)
    zero_src = torch.zeros_like(x0_source)

    hin_uu = torch.cat([x_t, zero_src], dim=1)
    hin_tu = torch.cat([x_t, zero_src], dim=1)
    hin_tt = torch.cat([x_t, x0_source], dim=1)

    # Stack into batch=3 for a single forward.
    h_cat = torch.cat([hin_uu, hin_tu, hin_tt], dim=0)
    t_cat = torch.cat([t_in, t_in, t_in], dim=0)
    c_cat = torch.cat([null_embeds, cond_embeds, cond_embeds], dim=0)
    v = _transformer_forward(transformer, h_cat, c_cat, t_cat)
    v_uu, v_tu, v_tt = v.chunk(3)
    return v_uu + cfg_text * (v_tu - v_uu) + cfg_image * (v_tt - v_tu)


@torch.no_grad()
def rfsolver_step(transformer, x_t, x0_source, cond_embeds, null_embeds,
                   t_cur, t_next, delta_norm, order: int,
                   cfg_mode: str, cfg_scale: float,
                   cfg_text: float, cfg_image: float):
    t_cur_f, t_next_f = float(t_cur), float(t_next)
    dt_norm = (t_next_f - t_cur_f) / 1000.0

    def _v(x, t_scalar):
        t_t = torch.tensor([t_scalar], device=x.device)
        if cfg_mode == "dual":
            return predict_velocity_dual_cfg(
                transformer, x, x0_source, cond_embeds, null_embeds,
                t_t, cfg_text, cfg_image,
            )
        return predict_velocity_text_cfg(
            transformer, x, x0_source, cond_embeds, null_embeds,
            t_t, cfg_scale,
        )

    v_cur = _v(x_t, t_cur_f)
    if order == 1:
        return x_t + dt_norm * v_cur

    sign = 1.0 if dt_norm >= 0 else -1.0
    delta_signed = delta_norm * sign
    delta_in_sched = delta_signed * 1000.0
    x_probe = x_t + delta_signed * v_cur
    t_probe_f = max(0.0, min(999.0, t_cur_f + delta_in_sched))
    v_probe = _v(x_probe, t_probe_f)
    v_deriv = (v_probe - v_cur) / delta_signed
    return x_t + dt_norm * v_cur + 0.5 * (dt_norm ** 2) * v_deriv


@torch.no_grad()
def denoise_from_noise(transformer, scheduler, x0_source,
                        cond_embeds, null_embeds,
                        num_steps: int, delta_norm: float, order: int,
                        cfg_mode: str, cfg_scale: float,
                        cfg_text: float, cfg_image: float,
                        device, init_noise=None,
                        desc: str = "RF-Solver denoise"):
    """xT = noise -> x0_edited, with x0_source concat-injected at every step."""
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_desc = scheduler.timesteps.tolist() + [0.0]

    if init_noise is None:
        x = torch.randn_like(x0_source)
    else:
        x = init_noise.to(x0_source.dtype).clone()

    for i in tqdm(range(len(timesteps_desc) - 1), desc=desc, leave=False):
        x = rfsolver_step(
            transformer, x, x0_source, cond_embeds, null_embeds,
            t_cur=timesteps_desc[i], t_next=timesteps_desc[i + 1],
            delta_norm=delta_norm, order=order,
            cfg_mode=cfg_mode, cfg_scale=cfg_scale,
            cfg_text=cfg_text, cfg_image=cfg_image,
        )
    return x


@torch.no_grad()
def rf_inversion(transformer, scheduler, x0_source,
                  null_embeds, num_steps: int, delta_norm: float, order: int,
                  device, desc: str = "RF-Solver inversion"):
    """Optional: x0_source -> xT under (null cond, x0_source in channel).
    Used only when --use_inversion is set, for deterministic editing.
    """
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_asc = scheduler.timesteps.flip(0).tolist() + [1000.0]
    x = x0_source.clone()
    for i in tqdm(range(len(timesteps_asc) - 1), desc=desc, leave=False):
        x = rfsolver_step(
            transformer, x, x0_source, null_embeds, null_embeds,
            t_cur=timesteps_asc[i], t_next=timesteps_asc[i + 1],
            delta_norm=delta_norm, order=order,
            cfg_mode="text", cfg_scale=1.0,
            cfg_text=1.0, cfg_image=1.0,
        )
    return x


# =============================================================================
# Visualisation
# =============================================================================

def to_pil(t):
    arr = t.squeeze(0).float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_grid(images, labels):
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    # ckpt + paths
    parser.add_argument("--finetuned_ckpt", type=str, required=True,
                         help="Stage 2 transformer.pt (must be 64-ch pos_embed). "
                              "editclip_adapter.pt is auto-loaded from the "
                              "same directory.")
    parser.add_argument("--editclip_path", type=str,
                         default="/root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14",
                         help="Same EditCLIP encoder used for Stage 2 training.")
    parser.add_argument("--nitroE_local_path", type=str, default=None)
    parser.add_argument("--dcae_local_path", type=str, default=None)
    # inputs
    parser.add_argument("--exemplar_input", type=str, required=True)
    parser.add_argument("--exemplar_edited", type=str, required=True)
    parser.add_argument("--query_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                         default="output/inference_ip2p_editclip")
    # RF-Solver knobs
    parser.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--delta_t", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=30,
                         help="Denoising steps (xT -> x0).")
    parser.add_argument("--num_inversion_steps", type=int, default=30,
                         help="Only used when --use_inversion.")
    # CFG
    parser.add_argument("--cfg_mode", type=str, default="text",
                         choices=["text", "dual"],
                         help="'text' (default) = single-CFG over text cond, "
                              "source always in channel. 1 fwd per step. "
                              "'dual' = IP2P paper dual-CFG with separate "
                              "text/image scales. 3 fwd per step.")
    parser.add_argument("--cfg_scale", type=float, default=4.5,
                         help="Used when --cfg_mode text.")
    parser.add_argument("--cfg_scale_list", type=float, nargs="+", default=None,
                         help="Sweep over text-CFG values. Source always in "
                              "channel.")
    parser.add_argument("--cfg_text", type=float, default=4.5,
                         help="Text CFG scale for --cfg_mode dual.")
    parser.add_argument("--cfg_image", type=float, default=1.5,
                         help="Image CFG scale for --cfg_mode dual.")
    parser.add_argument("--adapter_scale", type=float, default=1.0,
                         help="Optional multiplier on adapter output.")
    parser.add_argument("--cond_token_mode", type=str, default=None,
                         choices=[None, "patches", "cls"],
                         help="patches: 256 patch tokens (default for old "
                              "checkpoints). cls: CLS replicated to 4 tokens. "
                              "If omitted, auto-detected from the ckpt dir's "
                              "cond_token_mode.txt marker.")
    parser.add_argument("--keep_cls", action="store_true",
                         help="DEPRECATED. Use --cond_token_mode instead.")
    # mode: random vs inversion
    parser.add_argument("--use_inversion", action="store_true",
                         help="Instead of random xT, invert the source latent "
                              "under null cond + concat. Deterministic but "
                              "subject to Nitro-E's inversion limits.")
    # extras
    parser.add_argument("--also_reconstruction", action="store_true",
                         help="Also denoise under NULL cond (image cond still "
                              "present in channel) to verify the model can "
                              "reconstruct the source. Sanity check.")
    parser.add_argument("--also_no_source", action="store_true",
                         help="Also generate with x0_source replaced by "
                              "zeros — text-prompt-only generation, no "
                              "structural anchor. Tells you how much of the "
                              "edit identity comes from the source channel.")
    # misc
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bf16",
                         choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # ----- load -----
    transformer, adapter, vae, editclip_encoder, editclip_processor, scheduler = \
        load_models(
            args.finetuned_ckpt, args.editclip_path,
            args.resolution, device, dtype,
            nitroE_local_path=args.nitroE_local_path,
            dcae_local_path=args.dcae_local_path,
        )

    # Diagnostic: how trained is pos_embed extra-32ch?
    w = transformer.pos_embed.proj.weight.detach()
    half = w.shape[1] // 2
    first_abs = w[:, :half].abs().mean().item()
    extra_abs = w[:, half:].abs().mean().item()
    ratio = extra_abs / (first_abs + 1e-12)
    print(f"[diag] pos_embed.proj:  first {half}-ch mean_abs={first_abs:.4e}  "
          f"extra {half}-ch mean_abs={extra_abs:.4e}  ratio={ratio:.3f}")
    if ratio < 0.1:
        print("[diag] WARNING: extra-32ch barely trained — channel concat "
              "likely NOT functional. Output will ignore source.")

    # ----- inputs -----
    print(f"[input] exemplar_input : {args.exemplar_input}")
    print(f"[input] exemplar_edited: {args.exemplar_edited}")
    print(f"[input] query_image    : {args.query_image}")
    ex_in_pil, _ = load_image(args.exemplar_input, args.resolution)
    ex_ed_pil, _ = load_image(args.exemplar_edited, args.resolution)
    query_pil, query_tensor = load_image(args.query_image, args.resolution)

    # ----- EditCLIP cond -----
    # Resolve cond_token_mode in priority: explicit CLI arg > marker
    # file > default 'patches'.
    cond_token_mode = args.cond_token_mode
    if cond_token_mode is None:
        marker = os.path.join(os.path.dirname(args.finetuned_ckpt),
                                "cond_token_mode.txt")
        if os.path.exists(marker):
            with open(marker) as f:
                cond_token_mode = f.read().strip()
            print(f"[cond] cond_token_mode.txt -> '{cond_token_mode}' "
                  f"(auto-detected)")
        else:
            cond_token_mode = "patches"
            print(f"[cond] no cond_token_mode.txt marker, defaulting to "
                  f"'patches' (256 tokens)")
    print(f"[cond] EditCLIP+adapter -> edit_embeds "
          f"(adapter_scale={args.adapter_scale}, "
          f"cond_token_mode={cond_token_mode})")
    edit_embeds = compute_edit_embeds(
        ex_in_pil, ex_ed_pil,
        editclip_encoder, editclip_processor, adapter,
        device, dtype, adapter_scale=args.adapter_scale,
        cond_token_mode=cond_token_mode,
    )
    null_embeds = torch.zeros_like(edit_embeds)
    _ef = edit_embeds.float()
    print(f"[cond] edit_embeds {tuple(edit_embeds.shape)}  "
          f"std={_ef.std().item():.3f}  "
          f"per-token L2 mean={_ef[0].norm(dim=-1).mean().item():.2f}")

    # ----- query -> source latent (clean) -----
    x0_source = encode_to_latent(query_tensor, vae, device, dtype)
    vae_recon = decode_from_latent(x0_source, vae).cpu()
    vae_psnr = psnr(query_tensor, vae_recon)
    print(f"[ref] VAE-only PSNR={vae_psnr:.2f}  "
          f"(ceiling for full-preserve case)")

    # ----- xT: random OR inversion -----
    if args.use_inversion:
        print(f"\n[invert] RF-Solver inversion under null cond + concat")
        xT = rf_inversion(
            transformer, scheduler, x0_source,
            null_embeds=null_embeds,
            num_steps=args.num_inversion_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            device=device,
        )
        print(f"[invert] xT  mean={xT.float().mean().item():.3f}  "
              f"std={xT.float().std().item():.3f}")
    else:
        print(f"\n[noise] xT = random N(0, I)  (IP2P style)")
        g = torch.Generator(device=device).manual_seed(args.seed)
        xT = torch.randn(*x0_source.shape, generator=g,
                          device=device, dtype=dtype)

    # ----- grid prep -----
    out_pils = [ex_in_pil, ex_ed_pil, query_pil, to_pil(vae_recon)]
    out_labels = ["ex-in", "ex-ed", "query", "VAE-only"]
    results = {}

    # ----- optional reconstruction (null cond + source-in-channel) -----
    if args.also_reconstruction:
        print(f"\n[recon] null-cond denoise (source still in channel)")
        x0_recon = denoise_from_noise(
            transformer, scheduler, x0_source,
            cond_embeds=null_embeds, null_embeds=null_embeds,
            num_steps=args.num_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_mode="text", cfg_scale=1.0,
            cfg_text=1.0, cfg_image=1.0,
            device=device, init_noise=xT,
            desc="recon",
        )
        recon_pixels = decode_from_latent(x0_recon, vae).cpu()
        recon_pil = to_pil(recon_pixels)
        recon_psnr = psnr(query_tensor, recon_pixels)
        latent_mse = F.mse_loss(x0_source.float(), x0_recon.float()).item()
        print(f"[recon] PSNR={recon_psnr:.2f}  latentMSE={latent_mse:.5f}  "
              f"(VAE baseline: {vae_psnr:.2f})")
        out_pils.append(recon_pil)
        out_labels.append(f"recon\nPSNR={recon_psnr:.1f}")
        results["reconstruction"] = {
            "psnr": recon_psnr, "latent_mse": latent_mse,
            "vae_psnr_ref": vae_psnr,
        }

    # ----- optional source-ablated generation (sanity check) -----
    if args.also_no_source:
        print(f"\n[no_src] denoise with zeros in channel (source ablated)")
        x0_no_src = denoise_from_noise(
            transformer, scheduler, torch.zeros_like(x0_source),
            cond_embeds=edit_embeds, null_embeds=null_embeds,
            num_steps=args.num_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_mode="text",
            cfg_scale=args.cfg_scale_list[0] if args.cfg_scale_list else args.cfg_scale,
            cfg_text=args.cfg_text, cfg_image=args.cfg_image,
            device=device, init_noise=xT,
            desc="no_src",
        )
        no_src_pil = to_pil(decode_from_latent(x0_no_src, vae).cpu())
        out_pils.append(no_src_pil)
        out_labels.append("no_source")

    # ----- edit pass(es) -----
    if args.cfg_mode == "dual":
        cfg_list = [None]  # single dual-CFG run
    else:
        cfg_list = (args.cfg_scale_list
                     if args.cfg_scale_list is not None
                     else [args.cfg_scale])
    print(f"\n[edit] cfg_mode={args.cfg_mode}  "
          f"sweep={cfg_list if args.cfg_mode == 'text' else 'N/A'}")

    results["edit"] = []
    for cfg in cfg_list:
        if args.cfg_mode == "dual":
            tag = f"dual\ncfg_T={args.cfg_text}\ncfg_I={args.cfg_image}"
            print(f"\n[edit] dual-CFG: text={args.cfg_text}  "
                  f"image={args.cfg_image}")
            x0_edit = denoise_from_noise(
                transformer, scheduler, x0_source,
                cond_embeds=edit_embeds, null_embeds=null_embeds,
                num_steps=args.num_steps,
                delta_norm=args.delta_t, order=args.solver_order,
                cfg_mode="dual", cfg_scale=1.0,
                cfg_text=args.cfg_text, cfg_image=args.cfg_image,
                device=device, init_noise=xT,
                desc="edit",
            )
        else:
            tag = f"text\ncfg={cfg}"
            print(f"\n[edit] text-CFG cfg={cfg}")
            x0_edit = denoise_from_noise(
                transformer, scheduler, x0_source,
                cond_embeds=edit_embeds, null_embeds=null_embeds,
                num_steps=args.num_steps,
                delta_norm=args.delta_t, order=args.solver_order,
                cfg_mode="text", cfg_scale=cfg,
                cfg_text=1.0, cfg_image=1.0,
                device=device, init_noise=xT,
                desc=f"edit cfg={cfg}",
            )

        edit_pixels = decode_from_latent(x0_edit, vae).cpu()
        edit_pil = to_pil(edit_pixels)
        edit_psnr_vs_query = psnr(query_tensor, edit_pixels)
        print(f"[edit] PSNR vs query={edit_psnr_vs_query:.2f}  "
              f"(VAE-only ceiling: {vae_psnr:.2f})")

        out_pils.append(edit_pil)
        out_labels.append(tag)
        results["edit"].append({
            "cfg_mode": args.cfg_mode,
            "cfg_scale": cfg if args.cfg_mode == "text" else None,
            "cfg_text": args.cfg_text if args.cfg_mode == "dual" else None,
            "cfg_image": args.cfg_image if args.cfg_mode == "dual" else None,
            "psnr_vs_query": edit_psnr_vs_query,
        })
        edit_pil.save(os.path.join(
            args.output_dir,
            f"edit_order{args.solver_order}_{tag.replace(chr(10), '_')}.png",
        ))

    # ----- grid + summary -----
    grid = make_grid(out_pils, out_labels)
    cfg_tag = (f"cfg{'-'.join(str(c) for c in cfg_list)}"
                if args.cfg_mode == "text" else
                f"dual_t{args.cfg_text}_i{args.cfg_image}")
    grid.save(os.path.join(
        args.output_dir,
        f"grid_order{args.solver_order}_{cfg_tag}.png",
    ))

    summary = {
        "args": vars(args),
        "pos_embed_extra_first_ratio": ratio,
        "vae_only_psnr": vae_psnr,
        "results": results,
    }
    with open(os.path.join(args.output_dir,
                              f"summary_order{args.solver_order}.json"),
                "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
