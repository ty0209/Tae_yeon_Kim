#!/usr/bin/env python
# coding=utf-8
"""
RF-Solver inference for EditCLIP + E-MMDiT with **Stable Flow** training-free
editing (Avrahami et al. 2024) adapted to Nitro-E.

Why Stable Flow
---------------
E-MMDiT (304M) is small enough that with CFG > 1 it tends to drift away
from the source structure: null-conditioned inversion lands at x_T, but
edit-conditioned denoising from x_T pulls the latent strongly toward the
text/cond's prior, "forgetting" the source. Stable Flow tackles this with
THREE complementary interventions, all training-free:

  1. **Vital-layer protection**: identify DiT blocks that are critical
     for content generation. Run them normally on the EDIT pass — do not
     overwrite their attention. Only inject on NON-vital layers.

  2. **Non-vital layer Q/K injection**: For non-vital blocks, replay the
     source trajectory's self-attn Q/K (recorded during the null-cond
     source pass) into the edit pass. This nails spatial routing to the
     source — analogous to PnP feature injection, but layer-selective.

  3. **Latent nudging**: At each denoising step, gently pull the current
     latent toward the corresponding inversion-anchor latent. Small
     coefficient (default 0.1) so the edit still happens but the latent
     does not run away from the source trajectory.

Default vital_layers (Nitro-E specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nitro-E has 24 blocks split [4, 16, 4]. The middle 16 group is where
semantic content lives — most blocks in that group are vital. Defaults:

    vital_layers   = blocks [8, 9, 10, 11, 12, 13, 14, 15]  (deeper half of middle)
    inject_layers  = all others (= [0..7] + [16..23])

You can override via CLI. To find the right `vital_layers` set
empirically, run a recon-quality probe: ablate each block (zero its
attention output) one at a time and see which ones destroy the
reconstruction. Those are "vital". (Not automated here; provided as
manual recipe.)

Pipeline
--------
1. EditCLIP -> adapter -> edit_embeds (same as rfsolver.py)
2. RF-Solver inversion (null cond), SAVING per-step latent anchors AND
   per-(block, step) self-attn Q/K from the SOURCE recording pass.
3. Edit pass: edit cond + CFG, with
     - Q/K injection on NON-vital blocks
     - Latent nudging toward anchor[i] every step
4. Decode.

Knobs (new vs rfsolver.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~
  --vital_layers      INT[]   (default: 8..15)
  --inject_layers     INT[]   (default: complement of vital_layers)
  --nudge_scale       FLOAT   (default 0.1)
  --nudge_start_step  INT     (default 0)
  --nudge_stop_step   INT     (default = num_denoising_steps)
  --record_source_pass BOOL   (default True; needs anchors + Q/K cache)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Make sure ALL HF library calls are fully offline. Set these BEFORE any
# huggingface_hub / transformers / diffusers import so cache-only behaviour
# is enforced (no HEAD requests, no metadata refresh, no telemetry).
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
from transformers import AutoProcessor, CLIPModel

from core.models.transformer_emmdit import EMMDiTTransformer


# =============================================================================
# Model loading (identical to inference_editclip_emmdit.py)
# =============================================================================

def build_adapter_from_state(state):
    """Reconstruct an editclip_adapter from its saved state_dict, auto-detecting
    whether it's the legacy single-Linear+LN form or the new MLP form.

    Layout convention (train_editclip_emmdit2.py):
      linear : '0.*' = Linear,  '1.*' = LayerNorm
      mlp    : '0.*' = Linear,  '1' GELU(no params),
               '2.*' = Linear,  '3.*' = LayerNorm
    """
    keys = set(state.keys())
    has_layer2 = "2.weight" in keys
    if has_layer2:
        editclip_dim = state["0.weight"].shape[1]
        hidden_dim   = state["0.weight"].shape[0]
        caption_dim  = state["2.weight"].shape[0]
        assert state["2.weight"].shape[1] == hidden_dim, \
            (f"MLP adapter dim mismatch: layer-2 expects {hidden_dim}, got "
             f"{state['2.weight'].shape}")
        adapter = nn.Sequential(
            nn.Linear(editclip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, caption_dim),
            nn.LayerNorm(caption_dim),
        )
        print(f"[load] EditCLIP adapter [mlp]: "
              f"Linear({editclip_dim} -> {hidden_dim}) -> GELU -> "
              f"Linear({hidden_dim} -> {caption_dim}) + LN")
    else:
        editclip_dim = state["0.weight"].shape[1]
        caption_dim  = state["0.weight"].shape[0]
        adapter = nn.Sequential(
            nn.Linear(editclip_dim, caption_dim),
            nn.LayerNorm(caption_dim),
        )
        print(f"[load] EditCLIP adapter [linear]: "
              f"Linear({editclip_dim} -> {caption_dim}) + LN")
    adapter.load_state_dict(state)
    return adapter


def _find_local_dcae() -> Optional[str]:
    """Locate the DC-AE VAE in the standard HF cache. Returns the local
    snapshot dir if present, else None.
    """
    repo = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    cache_root = Path(os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if not cache_dir.is_dir():
        return None
    snaps = cache_dir / "snapshots"
    if not snaps.is_dir():
        return None
    # Pick the first snapshot that contains a model file.
    for sn in sorted(snaps.iterdir()):
        if (sn / "diffusion_pytorch_model.safetensors").exists() or \
           (sn / "config.json").exists():
            return str(sn.resolve())
    return None


def load_models(finetuned_ckpt, editclip_path, resolution, device, dtype,
                 dcae_local_path: Optional[str] = None):
    print(f"[load] E-MMDiT transformer (fine-tuned: {finetuned_ckpt})")
    transformer = EMMDiTTransformer(
        sample_size=resolution // 32,
        use_sub_attn=(resolution <= 512),
    )
    state = torch.load(finetuned_ckpt, map_location="cpu")
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    adapter_path = os.path.join(os.path.dirname(finetuned_ckpt), "editclip_adapter.pt")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"editclip_adapter.pt not found at {adapter_path}")
    adapter_state = torch.load(adapter_path, map_location="cpu")
    editclip_adapter = build_adapter_from_state(adapter_state)
    editclip_adapter = editclip_adapter.to(device, dtype=dtype).eval()

    # ---- DC-AE VAE: cache-first, local_files_only=True. No network. ----
    if dcae_local_path is None:
        dcae_local_path = _find_local_dcae()
    if dcae_local_path is None:
        # Fall back to HF id, but force offline mode. If not cached, this
        # raises an error rather than reaching out.
        dcae_src = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
        print(f"[load] DC-AE VAE (HF id, local_files_only=True): {dcae_src}")
    else:
        dcae_src = dcae_local_path
        print(f"[load] DC-AE VAE (local cache snapshot): {dcae_src}")
    vae = AutoencoderDC.from_pretrained(
        dcae_src, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()

    # ---- EditCLIP — always local path, also force local_files_only ----
    print(f"[load] EditCLIP encoder ({editclip_path})")
    editclip_full = CLIPModel.from_pretrained(
        editclip_path, local_files_only=True,
    )
    editclip_encoder = editclip_full.vision_model.to(device, dtype=dtype).eval()
    editclip_processor = AutoProcessor.from_pretrained(
        editclip_path, local_files_only=True,
    )
    del editclip_full

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    return transformer, editclip_adapter, vae, editclip_encoder, editclip_processor, scheduler


# =============================================================================
# Image / latent helpers
# =============================================================================

def load_image(path, resolution):
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
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
                         device, dtype, adapter_scale: float = 1.0):
    """6-channel concat -> EditCLIP -> drop CLS -> adapter -> [1, 256, 2048].

    The optional `adapter_scale` multiplier compensates for the magnitude
    mismatch between adapter output and the Llama-3.2-1B distribution
    Nitro-E was originally trained on (std≈2.18, per-token L2≈98).

    - Legacy 'linear' adapter (LN init=1.0): output std≈0.92, L2≈41.
      Set adapter_scale ≈ 2.4 to match Llama L2.
    - New 'mlp' adapter (LN init=2.4): output already matches Llama L2.
      Use adapter_scale=1.0 (default).

    See diagnose_editclip_emmdit_distribution.py for the magnitude probe.
    """
    o = processor(images=ex_in_pil, return_tensors="pt")
    e = processor(images=ex_ed_pil, return_tensors="pt")
    concat = torch.cat([o.pixel_values, e.pixel_values], dim=1).to(device, dtype=dtype)
    feats = encoder(concat)[0][:, 1:, :]   # drop CLS -> [1, 256, 1024]
    out = adapter(feats)                     # [1, 256, 2048]
    if adapter_scale != 1.0:
        out = out * adapter_scale
    return out


# =============================================================================
# Stable Flow controller — layer-selective self-attn Q/K injection
# =============================================================================
#
# Lifecycle:
#   1. install_hooks(transformer)   — once, before any inference
#   2. set_mode("record")           — record Q/K during source recording pass
#   3. set_step(i) inside loop
#   4. run source pass (null cond)  — cache fills up
#   5. set_mode("inject")           — replay Q/K during edit pass
#   6. set_step(i) inside loop
#   7. run edit pass (edit cond + CFG)
#   8. remove_hooks()
#
# Layer selection:
#   - vital_layers : block indices that are NEVER touched, even when inject
#                    mode is active. They run normally.
#   - inject_layers: block indices where Q/K is recorded (in record mode) and
#                    replayed (in inject mode). Default = all - vital_layers.
# =============================================================================

class StableFlowController:
    def __init__(self, vital_layers: set, inject_layers: set,
                 step_start: int, step_end: int,
                 inject_q: bool = True, inject_k: bool = True,
                 inject_branch: str = "both",
                 verbose: bool = False):
        self.vital_layers = set(vital_layers)
        self.inject_layers = set(inject_layers) - self.vital_layers
        self.step_start = step_start
        self.step_end = step_end
        self.inject_q = inject_q
        self.inject_k = inject_k
        assert inject_branch in {"both", "cond", "null"}, \
            f"inject_branch must be 'both' | 'cond' | 'null', got {inject_branch!r}"
        self.inject_branch = inject_branch
        self.verbose = verbose
        self.cache: dict = {}      # {block_idx: {step_idx: {"q": T, "k": T}}}
        self.mode = "off"           # "off" | "record" | "inject"
        self.step_idx = -1
        self.is_probe = False
        self.handles = []
        self._n_record = 0
        self._n_inject = 0

    def set_mode(self, mode: str):
        assert mode in {"off", "record", "inject"}
        self.mode = mode

    def set_step(self, step_idx: int, is_probe: bool = False):
        self.step_idx = step_idx
        self.is_probe = is_probe

    def reset_counters(self):
        self._n_record = 0
        self._n_inject = 0

    def in_window(self, block_idx: int) -> bool:
        if block_idx in self.vital_layers:
            return False
        if block_idx not in self.inject_layers:
            return False
        return self.step_start <= self.step_idx < self.step_end

    def _make_hook(self, block_idx: int, qk: str):
        def hook(module, inputs, output):
            if self.mode == "off":
                return output
            if not self.in_window(block_idx):
                return output

            if self.mode == "record":
                if self.is_probe:
                    return output
                slot = self.cache.setdefault(block_idx, {}).setdefault(
                    self.step_idx, {})
                slot[qk] = output.detach().clone()
                self._n_record += 1
                return output

            # inject
            if (qk == "q" and not self.inject_q) or \
               (qk == "k" and not self.inject_k):
                return output
            cached = self.cache.get(block_idx, {}).get(
                self.step_idx, {}).get(qk)
            if cached is None:
                return output
            B_out = output.shape[0]
            B_cache = cached.shape[0]
            factor = B_out // B_cache if B_cache > 0 else 1

            if self.inject_branch == "both" or factor == 1:
                # Replace the WHOLE output (legacy / non-CFG case).
                # When factor=1 (no CFG), branch choice is moot.
                if B_out != B_cache:
                    reps = [1] * cached.dim()
                    reps[0] = factor
                    cached_full = cached.repeat(*reps)
                else:
                    cached_full = cached
                self._n_inject += 1
                return cached_full.to(output.dtype)

            # CFG layout: rows are [null_part (size B_cache) || cond_part (size B_cache)].
            # Inject only on the chosen half so v_null and v_cond diverge.
            assert factor == 2, (
                f"inject_branch={self.inject_branch!r} expects CFG factor=2 "
                f"(null/cond split), got factor={factor}"
            )
            out = output.clone()
            if self.inject_branch == "cond":
                out[B_cache:] = cached.to(output.dtype)
            elif self.inject_branch == "null":
                out[:B_cache] = cached.to(output.dtype)
            self._n_inject += 1
            return out
        return hook

    def install_hooks(self, transformer) -> int:
        idx = 0
        for group in transformer.block_groups:
            for block in group:
                attn = block.attn
                self.handles.append(
                    attn.to_q.register_forward_hook(self._make_hook(idx, "q"))
                )
                self.handles.append(
                    attn.to_k.register_forward_hook(self._make_hook(idx, "k"))
                )
                idx += 1
        if self.verbose:
            print(f"[stable] hooked {idx} blocks. vital={sorted(self.vital_layers)} "
                  f"inject={sorted(self.inject_layers)}")
        return idx

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def cache_size_mb(self) -> float:
        b = 0
        for sm in self.cache.values():
            for qm in sm.values():
                for t in qm.values():
                    b += t.numel() * t.element_size()
        return b / (1024 ** 2)


# =============================================================================
# Vital-layer auto-detection (Stable Flow's key idea)
# =============================================================================
#
# For each block: temporarily zero its attention output (the image-stream
# residual it contributes), re-run a short denoise from the same xT under
# edit conditioning, and compare against an un-ablated reference. Blocks
# whose ablation produces the LARGEST output deviation are the "vital"
# ones — they carry the most semantic load.
#
# Top-K most-impactful blocks => vital_layers (these run normally on the
# edit pass). The rest are inject_layers (safe to overwrite with source Q/K).
# =============================================================================

class _AblationHook:
    """Forward hook that zeroes the image-stream attention output of a
    JointTransformerBlock's `attn` module. Joint attention returns
    (attn_output, context_attn_output); we zero the first and leave the
    text-side intact.
    """
    def __init__(self):
        self.handle = None

    def install(self, attn_module):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                img_out, ctx_out = output
                img_zero = torch.zeros_like(img_out)
                return (img_zero, ctx_out)
            return torch.zeros_like(output)
        self.handle = attn_module.register_forward_hook(hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _iter_blocks(transformer):
    """Yield (global_idx, block) for every transformer block in order."""
    idx = 0
    for group in transformer.block_groups:
        for block in group:
            yield idx, block
            idx += 1


@torch.no_grad()
def detect_vital_layers(transformer, scheduler, xT,
                         cond_embeds, null_embeds,
                         num_steps: int, delta_norm: float, cfg_scale: float,
                         device, n_vital: int = 8,
                         metric: str = "l2",
                         strategy: str = "top",
                         verbose: bool = True):
    """Auto-detect vital layers by per-block attention ablation.

    Args
    ----
    num_steps : reduced denoise length for the probe (e.g. 10 instead of 30)
                so the probe is fast. Vital-layer ranking is stable enough
                under short probes.
    n_vital   : how many blocks to mark as vital.
    metric    : "l2" (default) or "l1" — pixel-level distance between
                ablated and reference latents at t=0.
    strategy  : "top"    = HIGH-impact blocks become vital (Stable Flow paper
                          convention; vital layers run normally so edit
                          flows through them; non-vital get Q/K injection).
                "bottom" = LOW-impact blocks become vital (inverted; the
                          critical content-generation blocks get Q/K
                          injected = strongly locked to source. Use this
                          when edit strength is too high and you want
                          maximum preservation).

    Returns
    -------
    vital_set    : set[int] of block indices
    impacts      : list[(block_idx, impact)], sorted by impact desc
    """
    assert strategy in {"top", "bottom"}, \
        f"strategy must be 'top' or 'bottom', got {strategy!r}"
    n_total = sum(len(g) for g in transformer.block_groups)

    if verbose:
        print(f"[probe] vital-layer detection: {n_total} blocks, "
              f"{num_steps}-step probe, top-{n_vital} marked vital")

    # 1) Reference run with NO ablation.
    x0_ref = rfsolver_denoising(
        transformer, scheduler, xT,
        cond_embeds=cond_embeds, null_embeds=null_embeds,
        num_steps=num_steps, delta_norm=delta_norm,
        order=2, cfg_scale=cfg_scale, device=device,
        sf_controller=None, anchors_desc=None,
        nudge_scale=0.0, desc="probe-ref",
    )

    # 2) Per-block ablation runs.
    impacts = []
    blocks = list(_iter_blocks(transformer))
    for b_idx, block in tqdm(blocks, desc="vital-probe", leave=False):
        ah = _AblationHook()
        ah.install(block.attn)
        try:
            x0_i = rfsolver_denoising(
                transformer, scheduler, xT,
                cond_embeds=cond_embeds, null_embeds=null_embeds,
                num_steps=num_steps, delta_norm=delta_norm,
                order=2, cfg_scale=cfg_scale, device=device,
                sf_controller=None, anchors_desc=None,
                nudge_scale=0.0, desc=f"probe-b{b_idx}",
            )
            diff = (x0_ref - x0_i).float()
            if metric == "l1":
                score = diff.abs().mean().item()
            else:
                score = (diff ** 2).mean().sqrt().item()
            impacts.append((b_idx, score))
        finally:
            ah.remove()

    # 3) Rank and pick top-K or bottom-K based on strategy.
    impacts.sort(key=lambda x: x[1], reverse=True)  # desc by impact
    if strategy == "top":
        picked = impacts[:n_vital]
    else:  # "bottom"
        picked = impacts[-n_vital:]
    vital_set = set(b for b, _ in picked)

    if verbose:
        print(f"[probe] strategy = {strategy!r}  "
              f"({'HIGH-impact' if strategy == 'top' else 'LOW-impact'} "
              f"blocks marked vital)")
        print(f"[probe] block | impact (sorted desc)")
        for b, s in impacts:
            tag = " VITAL" if b in vital_set else ""
            print(f"        {b:3d}    | {s:.4e}{tag}")
        print(f"[probe] vital_layers = {sorted(vital_set)}")

    return vital_set, impacts


# =============================================================================
# RF-Solver core
# =============================================================================
#
# Rectified Flow ODE:  dZ_t = v_θ(Z_t, t) dt
#
# 2nd-order Taylor update (paper Eq. 8 for denoising / Eq. 9 for inversion):
#   Z_{t_next} = Z_t + (t_next - t)*v(Z_t, t)
#              + 0.5*(t_next - t)^2 * v'(Z_t, t)
#
# First-derivative v' estimated via finite difference (paper Algo 1):
#   Z_probe = Z_t + Δt * v(Z_t, t)
#   v_probe = v(Z_probe, t + Δt)
#   v'(Z_t, t) ≈ (v_probe - v(Z_t, t)) / Δt
#
# Setting order=1 reduces to the vanilla 1st-order Euler step (for ablation).
# =============================================================================

@torch.no_grad()
def predict_velocity_cfg(transformer, x, cond_embeds, null_embeds,
                          t_sched, cfg_scale, do_cfg):
    """One transformer forward. CFG combines null and cond branches if do_cfg."""
    batch = x.shape[0]
    t_in = t_sched.expand(batch).to(x.device)
    if do_cfg and cfg_scale > 1.0:
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
        v_un, v_co = v_all.chunk(2)
        return v_un + cfg_scale * (v_co - v_un)
    out = transformer(
        hidden_states=x,
        encoder_hidden_states=cond_embeds,
        timestep=t_in,
        return_dict=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.no_grad()
def rfsolver_step(transformer, x, cond_embeds, null_embeds,
                  t_cur, t_next, delta_norm, cfg_scale, order, do_cfg,
                  sf_controller=None, step_idx: int = 0):
    """One RF-Solver step. Optionally informs `sf_controller` of the
    current step and whether each forward is MAIN or PROBE so the Q/K
    cache slot is used consistently.
    """
    t_cur_f = float(t_cur)
    t_next_f = float(t_next)
    dt_norm = (t_next_f - t_cur_f) / 1000.0

    if sf_controller is not None:
        sf_controller.set_step(step_idx, is_probe=False)
    t_cur_t = torch.tensor([t_cur_f], device=x.device)
    v_cur = predict_velocity_cfg(transformer, x, cond_embeds, null_embeds,
                                  t_cur_t, cfg_scale, do_cfg)

    if order == 1:
        return x + dt_norm * v_cur

    sign = 1.0 if dt_norm >= 0 else -1.0
    delta_signed = delta_norm * sign
    delta_in_sched = delta_signed * 1000.0
    x_probe = x + delta_signed * v_cur
    t_probe_f = max(0.0, min(999.0, t_cur_f + delta_in_sched))
    t_probe_t = torch.tensor([t_probe_f], device=x.device)
    if sf_controller is not None:
        sf_controller.set_step(step_idx, is_probe=True)
    v_probe = predict_velocity_cfg(transformer, x_probe, cond_embeds, null_embeds,
                                    t_probe_t, cfg_scale, do_cfg)
    v_deriv = (v_probe - v_cur) / delta_signed
    return x + dt_norm * v_cur + 0.5 * (dt_norm ** 2) * v_deriv


@torch.no_grad()
def rfsolver_inversion(transformer, scheduler, x0, null_embeds, dummy_embeds,
                        num_steps, delta_norm, order, device,
                        return_anchors: bool = False):
    """Clean latent x_0 -> noisy latent x_T using RF-Solver under null cond.

    If return_anchors=True, also returns
       anchors (list of N+1 latents, one per timestep in timesteps_desc order
                so anchors[i] corresponds to denoising step i)
    """
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_asc = scheduler.timesteps.flip(0).tolist()
    timesteps_asc = timesteps_asc + [1000.0]

    x = x0.clone()
    inv_anchors = [x.clone()] if return_anchors else None
    for i in tqdm(range(len(timesteps_asc) - 1), desc="RF-Solver inversion",
                   leave=False):
        x = rfsolver_step(
            transformer, x, null_embeds, null_embeds,
            t_cur=timesteps_asc[i], t_next=timesteps_asc[i + 1],
            delta_norm=delta_norm, cfg_scale=1.0, order=order, do_cfg=False,
        )
        if return_anchors:
            inv_anchors.append(x.clone())
    if return_anchors:
        # Reverse so anchors[i] is what we'd want when denoising step i goes
        # from timesteps_desc[i] -> timesteps_desc[i+1].
        # timesteps_desc = timesteps_asc reversed. So anchor at timesteps_desc[i]
        # is the inv-anchor at timesteps_asc[N-i].
        anchors_desc = list(reversed(inv_anchors))
        return x, anchors_desc
    return x


@torch.no_grad()
def rfsolver_denoising(transformer, scheduler, xT, cond_embeds, null_embeds,
                        num_steps, delta_norm, order, cfg_scale, device,
                        sf_controller=None,
                        anchors_desc=None,
                        nudge_scale: float = 0.0,
                        nudge_start: int = 0, nudge_stop: int = None,
                        desc: str = "RF-Solver denoising"):
    """Noisy latent x_T -> clean latent x_0'.

    Stable Flow extensions:
      - sf_controller: StableFlowController for layer-selective Q/K injection
      - anchors_desc:  list of latents from inversion, aligned with
                       timesteps_desc[i] (anchors_desc[i] = inverted latent
                       at timesteps_desc[i]). Used for nudging.
      - nudge_scale:   pull current latent toward anchor every step:
                          x = (1 - alpha) * x + alpha * anchors_desc[i+1]
                       alpha=0 disables.
      - nudge_start/stop: step index window where nudging is active.
    """
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_desc = scheduler.timesteps.tolist() + [0.0]
    n_steps = len(timesteps_desc) - 1
    if nudge_stop is None:
        nudge_stop = n_steps

    x = xT.clone()
    do_cfg = cfg_scale > 1.0
    for i in tqdm(range(n_steps), desc=desc, leave=False):
        x = rfsolver_step(
            transformer, x, cond_embeds, null_embeds,
            t_cur=timesteps_desc[i], t_next=timesteps_desc[i + 1],
            delta_norm=delta_norm, cfg_scale=cfg_scale, order=order,
            do_cfg=do_cfg,
            sf_controller=sf_controller, step_idx=i,
        )
        # Latent nudging: pull toward the next inversion anchor.
        if (nudge_scale > 0 and anchors_desc is not None
                and nudge_start <= i < nudge_stop
                and i + 1 < len(anchors_desc)):
            target = anchors_desc[i + 1].to(x.dtype)
            x = (1.0 - nudge_scale) * x + nudge_scale * target
    return x


# =============================================================================
# Visualization
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
    parser = argparse.ArgumentParser(description=__doc__)
    # weights — all local; this script makes NO network calls.
    parser.add_argument("--finetuned_ckpt", type=str, required=True)
    parser.add_argument("--editclip_path", type=str,
                        default="/root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14")
    parser.add_argument("--dcae_local_path", type=str, default=None,
                        help="Optional explicit local path to the DC-AE VAE "
                             "directory. If None, looks in HF cache. "
                             "Either way the script runs in HF_HUB_OFFLINE=1.")
    # inputs
    parser.add_argument("--exemplar_input", type=str, required=True)
    parser.add_argument("--exemplar_edited", type=str, required=True)
    parser.add_argument("--query_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/inference_rfsolver")
    # RF-Solver knobs
    parser.add_argument("--solver_order", type=int, choices=[1, 2], default=2,
                        help="1 = vanilla RF (Euler), 2 = RF-Solver (Taylor). "
                             "Use 1 for ablation comparison.")
    parser.add_argument("--delta_t", type=float, default=0.01,
                        help="Normalized probe size for finite-difference derivative. "
                             "Paper recommends 0.01. Only used when solver_order=2.")
    parser.add_argument("--num_inversion_steps", type=int, default=30)
    parser.add_argument("--num_denoising_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=4.5,
                        help="Used when --cfg_scale_list is not provided.")
    parser.add_argument("--cfg_scale_list", type=float, nargs="+", default=None,
                        help="Sweep multiple CFG values in a single run. "
                             "Inversion is computed ONCE and reused; only the "
                             "denoising pass is repeated per CFG. Closest "
                             "analog to SDEdit's --strength_list sweep.")
    parser.add_argument("--adapter_scale", type=float, default=1.0,
                        help="Multiplier on the adapter output to compensate "
                             "for distribution magnitude mismatch with Llama "
                             "(Nitro-E's pretraining text encoder).")
    # ---- Stable Flow knobs ----
    parser.add_argument("--auto_detect_vital", action="store_true", default=True,
                        help="Run a short ablation probe to auto-detect "
                             "vital layers. Default ON; pass --no_auto_detect "
                             "to disable and use --vital_layers explicitly.")
    parser.add_argument("--no_auto_detect", dest="auto_detect_vital",
                        action="store_false")
    parser.add_argument("--n_vital", type=int, default=8,
                        help="When auto-detecting, number of blocks to mark "
                             "as vital. Default 8 (~1/3 of 24 blocks).")
    parser.add_argument("--vital_strategy", type=str, default="top",
                        choices=["top", "bottom"],
                        help="'top'    = HIGH-impact blocks are vital (paper "
                             "default; lets edit flow through them, locks "
                             "non-vital with Q/K injection). "
                             "'bottom' = LOW-impact blocks are vital (inverted; "
                             "the critical blocks get Q/K injected = locked "
                             "to source. Use when edit is too strong and "
                             "you want maximum preservation).")
    parser.add_argument("--probe_steps", type=int, default=10,
                        help="Number of denoise steps for the vital-layer "
                             "probe. Short ranking is stable; default 10 "
                             "(vs ~30 for real denoise) -> probe ~ 2-3 min.")
    parser.add_argument("--vital_layers", type=int, nargs="+",
                        default=None,
                        help="MANUAL override of vital layers. If provided, "
                             "skips auto-detection. Default None -> auto.")
    parser.add_argument("--inject_layers", type=int, nargs="+", default=None,
                        help="Block indices where Q/K is recorded + replayed. "
                             "If None, defaults to all blocks NOT in "
                             "--vital_layers. Use --inject_layers \"\" to "
                             "explicitly disable Q/K injection.")
    parser.add_argument("--no_qk_inject", action="store_true",
                        help="Disable Q/K injection entirely (Stable Flow w/o "
                             "the PnP-style branch). Useful for ablation.")
    parser.add_argument("--inject_branch", type=str, default="both",
                        choices=["both", "cond", "null"],
                        help="Which CFG branch to inject source Q/K into. "
                             "'both' (default, legacy) — image self-attn "
                             "contribution to v_cond-v_null = 0 (CFG weakened). "
                             "'cond' — only cond branch locked to source, "
                             "v_null is natural. "
                             "'null' — only null branch locked to source "
                             "(baseline pulls toward source), v_cond is "
                             "natural; CFG amplifies 'source -> cond' "
                             "direction. Often best for editing.")
    parser.add_argument("--nudge_scale", type=float, default=0.1,
                        help="Latent nudging strength. After each denoise "
                             "step, x = (1-alpha)*x + alpha*anchor[i+1]. "
                             "WARNING: this compounds across all 30 steps. "
                             "alpha=0.05 already pulls 79%% toward source "
                             "after 30 steps; alpha=0.1 pulls 96%%; alpha=0.2 "
                             "pulls 99.9%%; alpha=1.0 fully overwrites x with "
                             "the anchor every step (edit IMPOSSIBLE). "
                             "Practical range: 0.0 (off) to ~0.05. "
                             "Default 0.1 (already quite strong).")
    parser.add_argument("--nudge_start_step", type=int, default=0)
    parser.add_argument("--nudge_stop_step", type=int, default=None)
    parser.add_argument("--inject_start_step", type=int, default=0)
    parser.add_argument("--inject_stop_step", type=int, default=None)
    # also: reconstruction-only mode
    parser.add_argument("--also_reconstruction", action="store_true",
                        help="Additionally run a reconstruction pass (null "
                             "cond denoising) to measure preservation.")
    # misc
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    transformer, adapter, vae, editclip_encoder, editclip_processor, scheduler = load_models(
        args.finetuned_ckpt, args.editclip_path, args.resolution, device, dtype,
        dcae_local_path=args.dcae_local_path,
    )

    # Inputs
    print(f"[input] exemplar_input : {args.exemplar_input}")
    print(f"[input] exemplar_edited: {args.exemplar_edited}")
    print(f"[input] query_image    : {args.query_image}")
    ex_in_pil, _ = load_image(args.exemplar_input, args.resolution)
    ex_ed_pil, _ = load_image(args.exemplar_edited, args.resolution)
    query_pil, query_tensor = load_image(args.query_image, args.resolution)

    # Conditioning embeddings (EditCLIP -> drop CLS -> adapter -> [1, 256, 2048])
    print(f"[cond] computing EditCLIP -> adapter embeddings  "
          f"(adapter_scale={args.adapter_scale})")
    edit_embeds = compute_edit_embeds(
        ex_in_pil, ex_ed_pil,
        editclip_encoder, editclip_processor, adapter, device, dtype,
        adapter_scale=args.adapter_scale,
    )
    null_embeds = torch.zeros_like(edit_embeds)
    # Quick distribution sanity print (target: Llama L2≈98, std≈2.18)
    _ef = edit_embeds.float()
    print(f"[cond] edit_embeds shape: {tuple(edit_embeds.shape)}  "
          f"std={_ef.std().item():.3f}  "
          f"per-token L2 mean={_ef[0].norm(dim=-1).mean().item():.2f}")

    # Query -> latent
    x0 = encode_to_latent(query_tensor, vae, device, dtype)
    vae_recon = decode_from_latent(x0, vae).cpu()
    vae_psnr = psnr(query_tensor, vae_recon)
    print(f"[ref] VAE-only PSNR={vae_psnr:.2f} (ceiling for full-preserve case)")

    # -------------------------------------------------------------------------
    # Step 1: RF-Solver inversion (null-conditioned) — save anchors for nudging
    # -------------------------------------------------------------------------
    print(f"\n[invert] RF-Solver order={args.solver_order}, "
          f"steps={args.num_inversion_steps}, delta_t={args.delta_t}  "
          f"(save anchors for nudging)")
    xT, anchors_desc = rfsolver_inversion(
        transformer, scheduler, x0,
        null_embeds=null_embeds, dummy_embeds=null_embeds,
        num_steps=args.num_inversion_steps,
        delta_norm=args.delta_t,
        order=args.solver_order,
        device=device,
        return_anchors=True,
    )
    print(f"[invert] xT  mean={xT.float().mean().item():.3f}  "
          f"std={xT.float().std().item():.3f}   (target ~ N(0, 1))")
    print(f"[invert] {len(anchors_desc)} anchors saved")

    # -------------------------------------------------------------------------
    # Step 1.5: Install Stable Flow controller + run SOURCE RECORDING pass.
    #   - Resolve vital / inject layer sets
    #   - Run null-cond denoise to fill Q/K cache
    #   - The reconstruction PSNR (vs query) doubles as a sanity check
    # -------------------------------------------------------------------------
    out_pils = [ex_in_pil, ex_ed_pil, query_pil, to_pil(vae_recon)]
    out_labels = ["ex-in", "ex-ed", "query", "VAE-only"]
    results = {}

    n_total_blocks = sum(len(g) for g in transformer.block_groups)

    # ---- Resolve vital_layers: manual override OR auto-detection ----
    if args.vital_layers is not None and len(args.vital_layers) > 0:
        vital_set = set(args.vital_layers)
        print(f"[stable] manual --vital_layers = {sorted(vital_set)}")
    elif args.auto_detect_vital:
        # Use the first cfg in the sweep for the probe.
        probe_cfg = (args.cfg_scale_list[0]
                     if args.cfg_scale_list is not None
                     else args.cfg_scale)
        print(f"\n[probe] auto-detecting vital layers "
              f"(cfg={probe_cfg}, steps={args.probe_steps}, "
              f"{args.vital_strategy}-{args.n_vital})...")
        vital_set, vital_impacts = detect_vital_layers(
            transformer, scheduler, xT,
            cond_embeds=edit_embeds, null_embeds=null_embeds,
            num_steps=args.probe_steps,
            delta_norm=args.delta_t,
            cfg_scale=probe_cfg,
            device=device,
            n_vital=args.n_vital,
            metric="l2",
            strategy=args.vital_strategy,
            verbose=True,
        )
        results["vital_probe"] = {
            "n_vital": args.n_vital,
            "strategy": args.vital_strategy,
            "probe_steps": args.probe_steps,
            "probe_cfg": probe_cfg,
            "vital_layers": sorted(vital_set),
            "impacts": [{"block": int(b), "impact": float(s)}
                         for b, s in vital_impacts],
        }
    else:
        # Neither flag set — fall back to a reasonable hardcoded default.
        vital_set = set(range(8, 16))
        print(f"[stable] no auto-detect and no --vital_layers — "
              f"falling back to default {sorted(vital_set)}")

    if args.inject_layers is None or args.inject_layers == []:
        inject_set = set(range(n_total_blocks)) - vital_set
    else:
        inject_set = set(args.inject_layers) - vital_set
    if args.no_qk_inject:
        inject_set = set()

    inject_stop = (args.inject_stop_step if args.inject_stop_step is not None
                   else args.num_denoising_steps)
    sf_controller = StableFlowController(
        vital_layers=vital_set,
        inject_layers=inject_set,
        step_start=args.inject_start_step,
        step_end=inject_stop,
        inject_q=True, inject_k=True,
        inject_branch=args.inject_branch,
        verbose=True,
    )
    sf_controller.install_hooks(transformer)
    print(f"[stable] {n_total_blocks} blocks total; vital={sorted(vital_set)}; "
          f"inject={sorted(inject_set)}; inject_branch={args.inject_branch!r}")

    # ---- Source recording pass (null cond) — fills the Q/K cache ----
    if len(inject_set) > 0:
        sf_controller.set_mode("record")
        sf_controller.reset_counters()
        print(f"\n[source] null-cond denoise (recording Q/K)")
        x0_recon = rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=null_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=1.0,
            device=device,
            sf_controller=sf_controller,
            anchors_desc=None,    # no nudging during source pass
            nudge_scale=0.0,
            desc="src-rec",
        )
        sf_controller.set_mode("off")
        print(f"[stable] cached {sf_controller._n_record} Q/K tensors "
              f"({sf_controller.cache_size_mb():.1f} MB)")
    elif args.also_reconstruction:
        # No injection, but user asked for recon
        print(f"\n[recon] null-cond denoise (no injection requested)")
        x0_recon = rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=null_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=1.0,
            device=device,
        )
    else:
        x0_recon = None

    if x0_recon is not None:
        recon_pixels = decode_from_latent(x0_recon, vae).cpu()
        recon_pil = to_pil(recon_pixels)
        recon_psnr = psnr(query_tensor, recon_pixels)
        latent_mse = F.mse_loss(x0.float(), x0_recon.float()).item()
        print(f"[recon] PSNR={recon_psnr:.2f}  latentMSE={latent_mse:.5f}  "
              f"(VAE baseline: {vae_psnr:.2f})")
        if args.also_reconstruction:
            out_pils.append(recon_pil)
            out_labels.append(f"recon\nPSNR={recon_psnr:.1f}")
        results["reconstruction"] = {
            "psnr": recon_psnr, "latent_mse": latent_mse,
            "vae_psnr_ref": vae_psnr,
        }

    # -------------------------------------------------------------------------
    # Step 2b: Editing denoise from xT — sweep over one or more CFG values.
    # We reuse the same xT (inversion is the expensive part) and only re-run
    # the denoising pass per CFG.
    # -------------------------------------------------------------------------
    cfg_list = (args.cfg_scale_list
                if args.cfg_scale_list is not None
                else [args.cfg_scale])
    print(f"\n[edit] sweeping cfg_scale over {cfg_list}")

    results["edit"] = []
    nudge_stop = (args.nudge_stop_step if args.nudge_stop_step is not None
                  else args.num_denoising_steps)
    for cfg in cfg_list:
        if len(inject_set) > 0:
            sf_controller.set_mode("inject")
            sf_controller.reset_counters()
        print(f"\n[edit] cfg={cfg}  nudge={args.nudge_scale}  "
              f"inject_layers={len(inject_set)}  vital={len(vital_set)}")
        x0_edit = rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=edit_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=cfg,
            device=device,
            sf_controller=sf_controller if len(inject_set) > 0 else None,
            anchors_desc=anchors_desc,
            nudge_scale=args.nudge_scale,
            nudge_start=args.nudge_start_step,
            nudge_stop=nudge_stop,
            desc=f"edit cfg={cfg}",
        )
        if len(inject_set) > 0:
            sf_controller.set_mode("off")
            print(f"[stable] cfg={cfg}: {sf_controller._n_inject} injections")

        edit_pixels = decode_from_latent(x0_edit, vae).cpu()
        edit_pil = to_pil(edit_pixels)
        edit_psnr_vs_query = psnr(query_tensor, edit_pixels)
        print(f"[edit] cfg={cfg}  PSNR vs query={edit_psnr_vs_query:.2f}  "
              f"(VAE-only ceiling: {vae_psnr:.2f})")

        out_pils.append(edit_pil)
        tag = f"sf\ncfg={cfg}\nnudge={args.nudge_scale}"
        out_labels.append(tag)
        results["edit"].append({
            "cfg_scale": cfg,
            "psnr_vs_query": edit_psnr_vs_query,
            "vital_layers": sorted(vital_set),
            "inject_layers": sorted(inject_set),
            "nudge_scale": args.nudge_scale,
        })

        edit_pil.save(os.path.join(
            args.output_dir,
            f"edit_order{args.solver_order}_cfg{cfg}_nudge{args.nudge_scale}.png",
        ))

    # Clean up hooks after the sweep.
    sf_controller.remove_hooks()

    # Save recon alone (if computed)
    if args.also_reconstruction:
        to_pil(recon_pixels).save(os.path.join(
            args.output_dir, f"recon_order{args.solver_order}.png",
        ))

    # One big grid with every CFG result side by side
    cfg_tag = ("cfg" + "_".join(str(c) for c in cfg_list)
               if len(cfg_list) > 1
               else f"cfg{cfg_list[0]}")
    grid = make_grid(out_pils, out_labels)
    grid.save(os.path.join(
        args.output_dir,
        f"grid_order{args.solver_order}_{cfg_tag}.png",
    ))

    summary = {
        "args": vars(args),
        "edit_embeds_shape": list(edit_embeds.shape),
        "vae_only_psnr": vae_psnr,
        "results": results,
    }
    with open(os.path.join(args.output_dir,
                            f"summary_order{args.solver_order}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.output_dir}")
    print("       grid layout: [ex-in | ex-ed | query | VAE-only |", end="")
    if args.also_reconstruction:
        print(" recon |", end="")
    print(" edit_cfg=" + " | edit_cfg=".join(str(c) for c in cfg_list) + "]")


if __name__ == "__main__":
    main()
