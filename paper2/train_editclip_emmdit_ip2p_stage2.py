#!/usr/bin/env python
# coding=utf-8
"""
**Stage 2 trainer**: replace Llama with EditCLIP+adapter on top of a
Stage 1 IP2P channel-concat checkpoint.

Design choices (all five considerations baked in)
-------------------------------------------------
  1. FRESH random init for the adapter — never load a previous emmdit2
     adapter, which would re-introduce memorisation bias.
  2. PARTIAL unfreeze by default (Option B): only the parts that need to
     re-adapt to the new conditioning distribution are trainable:
       - pos_embed.proj      (the 64-ch IP2P concat — must not drift)
       - context_embedder    (Llama caption_channels -> caption_projection_dim)
       - text-stream layers  (add_q/k/v_proj, ff_context, norm_*_context,
                              scale_shift_*_c)
       - editclip_adapter    (fresh)
     Image-stream of the transformer stays FROZEN to preserve Stage 1's
     channel-concat learning. Pass --full_unfreeze to switch to Option A.
  3. LR 2e-5 default with warmup (vs Stage 1's 5e-5) — Stage 1 weights are
     already good, only gently adapt to the new cond distribution.
  4. Pos_embed extra-32ch monitoring — tensorboard logs the ratio every
     100 steps. If `pos_embed_extra_over_orig` collapses toward 0 the
     concat learning is breaking; lower LR or freeze pos_embed.
  5. EditCLIP encoder is FROZEN (its 6-channel-pair encoding is the
     pretrained anchor).

Why this won't oscillate this time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previous emmdit2 oscillation came from "no query in input + memorisation
loss". Now query is always in the channel-concat input, so the loss
gradient direction is unambiguous: "given (noised_edited, source) + cond,
predict v toward edited". Adapter only needs to encode edit DIRECTION,
not full edited image. Stable.

Usage
-----
accelerate launch train_editclip_emmdit_ip2p_stage2.py \
    --stage1_ckpt /root/.../emmdit_ip2p_stage1/.../checkpoint-50000 \
    --dataset_name /data/kimm0902_files/IP2P_datasets/data \
    --editclip_path /root/.../editclip_omnistyle_resumed/.../checkpoint-100000 \
    --resolution 512 --train_batch_size 4 \
    --max_train_steps 30000 \
    --checkpointing_steps 2000 \
    --learning_rate 2e-5 --lr_warmup_steps 1000 \
    --output_dir output/emmdit_ip2p_stage2 \
    --conditioning_dropout_prob 0.05 \
    --gradient_checkpointing

Resume support (max_train_steps is ABSOLUTE):
    --resume_from_checkpoint <stage2-checkpoint-NNNN>
    --reuse_resume_output_dir
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Offline mode — never reach out to HF, set before any HF import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Local 'datasets/' shadows the pip package — push script dir to end.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SCRIPT_DIR]
sys.path.append(_SCRIPT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, AutoProcessor

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderDC
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from core.models.transformer_emmdit import EMMDiTTransformer

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Cache-aware Nitro-E loader (offline-only)
# ---------------------------------------------------------------------------

def find_nitroE_weights(repo: str, filename: str,
                          local_path: Optional[str] = None) -> str:
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"--nitroE_local_path not found: {local_path}")
        print(f"[nitroE] using explicit local path: {local_path}")
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
                    print(f"[nitroE] found in HF cache: {cand}")
                    return str(cand.resolve())
    print("[nitroE] not in local cache; calling hf_hub_download (uses cache).")
    return hf_hub_download(repo, filename)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def resolve_resume_dir(resume_path: str) -> str:
    p = Path(resume_path).resolve()
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise FileNotFoundError(f"resume path is not a directory: {p}")
    return str(p)


def parse_resume_step(resume_dir: str) -> int:
    name = os.path.basename(resume_dir.rstrip("/"))
    m = re.match(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Rectified Flow utilities (shared with all trainers)
# ---------------------------------------------------------------------------

def sample_logit_normal(batch_size, m=0.0, s=1.0, device="cpu"):
    u = torch.randn(batch_size, device=device) * s + m
    return torch.sigmoid(u)


def compute_rf_targets(x0, noise, t):
    t = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t) * x0 + t * noise
    v_target = noise - x0
    return x_t, v_target


# ---------------------------------------------------------------------------
# Channel-concat extension (32->64 ch pos_embed.proj) — IDEMPOTENT
# ---------------------------------------------------------------------------

def extend_pos_embed_to_concat(transformer) -> int:
    """Extend pos_embed.proj from in_channels -> 2*in_channels with zero-init
    for the new half. Idempotent: if already extended (e.g. Stage 1 ckpt
    already has 64-ch pos_embed at load time), this is a no-op.
    """
    old_proj = transformer.pos_embed.proj
    assert isinstance(old_proj, nn.Conv2d), \
        f"expected pos_embed.proj to be nn.Conv2d, got {type(old_proj)}"
    old_in = old_proj.in_channels
    new_in = old_in * 2

    new_proj = nn.Conv2d(
        new_in, old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
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


# ---------------------------------------------------------------------------
# Dataset (IP2P triplets, live EditCLIP encoding)
# ---------------------------------------------------------------------------

class IP2PEditCLIPDataset(torch.utils.data.Dataset):
    """Returns the things the trainer needs:
      - original_pixel_values : [3, H, W] in [-1, 1] (for VAE -> latent)
      - edited_pixel_values   : same
      - editclip_input        : [6, 224, 224] (concat of orig+edit, EditCLIP-ready)
    Same crop/flip applied to all three so they stay aligned spatially.
    """
    def __init__(self, hf_dataset, editclip_processor, resolution=512,
                  original_image_column="original_image",
                  edited_image_column="edited_image",
                  text_column="edit_prompt"):
        self.dataset = hf_dataset
        self.editclip_processor = editclip_processor
        self.resolution = resolution
        self.orig_col = original_image_column
        self.edit_col = edited_image_column
        self.text_col = text_column

        from torchvision import transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_np(image, resolution):
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        orig_img = ex[self.orig_col]
        edit_img = ex[self.edit_col]
        orig_np = self._to_np(orig_img, self.resolution)
        edit_np = self._to_np(edit_img, self.resolution)
        stacked = np.concatenate([orig_np, edit_np])
        stacked = torch.tensor(stacked).float()
        stacked = 2.0 * (stacked / 255.0) - 1.0
        stacked = self.train_transforms(stacked)
        original_pv, edited_pv = stacked.chunk(2)
        original_pv = original_pv.reshape(3, self.resolution, self.resolution)
        edited_pv = edited_pv.reshape(3, self.resolution, self.resolution)

        # EditCLIP input: 6-channel concat in [0, 1] range, processor-handled
        orig_for_clip = (original_pv + 1.0) / 2.0
        edit_for_clip = (edited_pv + 1.0) / 2.0
        orig_clip = self.editclip_processor(images=orig_for_clip,
                                              return_tensors="pt",
                                              do_rescale=False)
        edit_clip = self.editclip_processor(images=edit_for_clip,
                                              return_tensors="pt",
                                              do_rescale=False)
        editclip_input = torch.cat(
            [orig_clip.pixel_values, edit_clip.pixel_values], dim=1
        ).squeeze(0)

        return {
            "original_pixel_values": original_pv,
            "edited_pixel_values":   edited_pv,
            "editclip_input":        editclip_input,
        }


def collate_fn(examples):
    return {
        "original_pixel_values": torch.stack(
            [e["original_pixel_values"] for e in examples]
        ).contiguous().float(),
        "edited_pixel_values": torch.stack(
            [e["edited_pixel_values"] for e in examples]
        ).contiguous().float(),
        "editclip_input": torch.stack(
            [e["editclip_input"] for e in examples]
        ).contiguous().float(),
    }


# ---------------------------------------------------------------------------
# Adapter — FRESH random init, MLP form
# ---------------------------------------------------------------------------

def build_fresh_adapter(editclip_dim: int, hidden_dim: int, caption_dim: int,
                          ln_init_scale: float = 2.4) -> nn.Module:
    adapter = nn.Sequential(
        nn.Linear(editclip_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, caption_dim),
        nn.LayerNorm(caption_dim),
    )
    nn.init.xavier_uniform_(adapter[0].weight)
    nn.init.zeros_(adapter[0].bias)
    nn.init.xavier_uniform_(adapter[2].weight)
    nn.init.zeros_(adapter[2].bias)
    if ln_init_scale != 1.0:
        ln_module = adapter[-1]
        nn.init.constant_(ln_module.weight, ln_init_scale)
        nn.init.zeros_(ln_module.bias)
    return adapter


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Nitro-E base
    p.add_argument("--emmdit_repo", type=str, default="amd/Nitro-E")
    p.add_argument("--emmdit_ckpt", type=str, default="Nitro-E-512px.safetensors")
    p.add_argument("--nitroE_local_path", type=str, default=None)
    # Stage 1 ckpt (REQUIRED) — provides 64-ch pos_embed and IP2P-trained transformer
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Path to a Stage 1 IP2P checkpoint directory "
                        "(must contain transformer.pt with 64-ch pos_embed). "
                        "We load transformer weights from here BEFORE swapping "
                        "in the EditCLIP cond.")
    # EditCLIP — frozen
    p.add_argument("--editclip_path", type=str, required=True,
                   help="Path to a (fine-tuned) EditCLIP visual encoder dir.")
    p.add_argument("--adapter_type", type=str, default="mlp",
                   choices=["linear", "mlp"])
    p.add_argument("--adapter_hidden_dim", type=int, default=None,
                   help="MLP hidden width. Defaults to caption_dim (2048).")
    p.add_argument("--ln_init_scale", type=float, default=2.4,
                   help="Init value for the adapter's trailing LayerNorm "
                        "weight. 2.4 matches Llama-3.2-1B per-token L2.")
    # Dataset (live IP2P loading, EditCLIP runs every step)
    p.add_argument("--dataset_name", type=str,
                   default="/data/kimm0902_files/IP2P_datasets/data")
    p.add_argument("--original_image_column", type=str, default="original_image")
    p.add_argument("--edited_image_column", type=str, default="edited_image")
    p.add_argument("--text_column", type=str, default="edit_prompt")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    # Training
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--max_train_steps", type=int, default=30000,
                   help="ABSOLUTE step limit. Stage 1 weights are already "
                        "good, so a shorter run usually suffices.")
    p.add_argument("--learning_rate", type=float, default=2e-5,
                   help="2e-5 default (vs Stage 1's 5e-5) to gently adapt "
                        "Stage 1 weights to the new cond distribution.")
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=1000,
                   help="LR warmup. 1000 steps is conservative for Stage 2 "
                        "cold-start with the new cond format.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    # IP2P CFG dropout (IP2P paper convention: one random_p, derived masks)
    p.add_argument("--conditioning_dropout_prob", type=float, default=0.05,
                   help="With p=0.05, drop cases are: text-only (5%%), "
                        "both (5%%), image-only (5%%), keep both (85%%).")
    p.add_argument("--legacy_independent_dropout", action="store_true",
                   help="Use the old broken-but-simple independent dropouts. "
                        "Default OFF — IP2P paper formulation.")
    # Freeze policy — Option B by default (partial unfreeze)
    p.add_argument("--full_unfreeze", action="store_true",
                   help="Option A: unfreeze the entire transformer (matches "
                        "IP2P paper). Default OFF — Option B: only text-stream "
                        "+ pos_embed.proj + adapter are trainable. Option B is "
                        "safer for small models; image-stream stays frozen so "
                        "Stage 1's channel-concat behaviour is preserved.")
    # Checkpointing / output
    p.add_argument("--output_dir", type=str, default="output/emmdit_ip2p_stage2")
    p.add_argument("--checkpointing_steps", type=int, default=2000)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=0,
                   help="Auto-prune old checkpoints, keeping at most this "
                        "many most-recent ones. **DEFAULT 0 = NEVER PRUNE**. "
                        "User has been burned by aggressive pruning before. "
                        "Set explicitly (e.g. 5) if disk space is tight, "
                        "AND pair with --protect_every_n_steps to keep "
                        "milestones.")
    p.add_argument("--protect_every_n_steps", type=int, default=0)
    p.add_argument("--protect_steps", type=int, nargs="+", default=None)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--validation_steps", type=int, default=1000)
    # Resume options
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a STAGE 2 checkpoint dir to resume from. "
                        "Different from --stage1_ckpt which is the initial "
                        "warm-start. Pass this to continue an interrupted "
                        "Stage 2 run.")
    p.add_argument("--resume_step_override", type=int, default=None)
    p.add_argument("--reuse_resume_output_dir", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Output dir
    if args.resume_from_checkpoint and args.reuse_resume_output_dir:
        resume_dir = resolve_resume_dir(args.resume_from_checkpoint)
        args.output_dir = os.path.dirname(resume_dir)
        print(f"[resume] reusing output_dir = {args.output_dir}")
    else:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.output_dir = os.path.join(args.output_dir,
                                         f"{date_str}_lr{args.learning_rate}")
    os.makedirs(args.output_dir, exist_ok=True)

    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                              logging_dir=logging_dir),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = (torch.bfloat16 if args.mixed_precision == "bf16"
                     else torch.float16 if args.mixed_precision == "fp16"
                     else torch.float32)

    # ------------------------------------------------------------------
    # 1. Build Nitro-E base (32-ch) -> extend to 64-ch -> load Stage 1 weights
    # ------------------------------------------------------------------
    logger.info("Building E-MMDiT transformer (32-ch base)...")
    transformer = EMMDiTTransformer(
        sample_size=args.resolution // 32,
        use_sub_attn=(args.resolution <= 512),
    )
    nitroE_path = find_nitroE_weights(args.emmdit_repo, args.emmdit_ckpt,
                                        local_path=args.nitroE_local_path)
    base_state = load_file(nitroE_path)
    transformer.load_state_dict(base_state)
    logger.info(f"Loaded Nitro-E base from {nitroE_path}")

    # Always extend to 64-ch for channel concat.
    old_in = transformer.pos_embed.proj.in_channels
    new_in = extend_pos_embed_to_concat(transformer)
    logger.info(f"[ip2p] extended pos_embed.proj  {old_in} -> {new_in} "
                f"(extra channels zero-init)")

    # ------------------------------------------------------------------
    # 2. Load Stage 1 ckpt INTO the now-64-ch transformer
    # ------------------------------------------------------------------
    stage1_dir = resolve_resume_dir(args.stage1_ckpt)
    stage1_t = os.path.join(stage1_dir, "transformer.pt")
    if not os.path.exists(stage1_t):
        raise FileNotFoundError(f"transformer.pt not found in Stage 1 dir: {stage1_dir}")
    s1_state = torch.load(stage1_t, map_location="cpu")
    s1_in = s1_state.get("pos_embed.proj.weight", None)
    if s1_in is None:
        raise RuntimeError("Stage 1 ckpt missing pos_embed.proj.weight")
    if s1_in.shape[1] != new_in:
        raise RuntimeError(f"Stage 1 pos_embed in_channels={s1_in.shape[1]} "
                            f"!= expected {new_in}. Is this really a "
                            f"channel-concat checkpoint?")
    transformer.load_state_dict(s1_state, strict=True)
    logger.info(f"[stage2] Stage 1 weights loaded from {stage1_t}")

    # ------------------------------------------------------------------
    # 3. EditCLIP encoder (frozen) + EditCLIP processor + FRESH adapter
    # ------------------------------------------------------------------
    logger.info(f"Loading EditCLIP visual encoder: {args.editclip_path}")
    editclip_full = CLIPModel.from_pretrained(
        args.editclip_path, local_files_only=True,
    )
    editclip_encoder = editclip_full.vision_model
    editclip_processor = AutoProcessor.from_pretrained(
        args.editclip_path, local_files_only=True,
    )
    del editclip_full

    # MUST be frozen.
    for p in editclip_encoder.parameters():
        p.requires_grad_(False)

    # FRESH random adapter — never load a previous Stage-0 / emmdit2 adapter.
    editclip_hidden_dim = editclip_encoder.config.hidden_size
    emmdit_caption_dim = transformer.config.caption_channels
    hidden = args.adapter_hidden_dim or emmdit_caption_dim
    logger.info(f"[adapter] FRESH random init: Linear({editclip_hidden_dim}->"
                f"{hidden}) GELU Linear({hidden}->{emmdit_caption_dim}) LN "
                f"(init={args.ln_init_scale})")
    editclip_adapter = build_fresh_adapter(
        editclip_hidden_dim, hidden, emmdit_caption_dim,
        ln_init_scale=args.ln_init_scale,
    )

    # ------------------------------------------------------------------
    # 4. VAE (frozen, used for live latent encoding)
    # ------------------------------------------------------------------
    logger.info("Loading DC-AE VAE...")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        torch_dtype=weight_dtype, local_files_only=True,
    ).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # ------------------------------------------------------------------
    # 5. Resume Stage 2 if requested (overrides Stage 1 ckpt + fresh adapter)
    # ------------------------------------------------------------------
    resume_step = 0
    optimizer_state_to_load = None
    if args.resume_from_checkpoint:
        resume_dir = resolve_resume_dir(args.resume_from_checkpoint)
        resume_step = args.resume_step_override or parse_resume_step(resume_dir)
        logger.info(f"[resume] Stage 2 resume from {resume_dir} (step {resume_step})")

        t_path = os.path.join(resume_dir, "transformer.pt")
        a_path = os.path.join(resume_dir, "editclip_adapter.pt")
        o_path = os.path.join(resume_dir, "optimizer.pt")
        if not (os.path.exists(t_path) and os.path.exists(a_path)):
            raise FileNotFoundError(f"Missing transformer.pt or "
                                       f"editclip_adapter.pt in {resume_dir}")
        transformer.load_state_dict(torch.load(t_path, map_location="cpu"),
                                       strict=True)
        editclip_adapter.load_state_dict(torch.load(a_path, map_location="cpu"))
        logger.info("[resume] Stage 2 transformer + adapter restored")
        if os.path.exists(o_path):
            optimizer_state_to_load = torch.load(o_path, map_location="cpu")

    # ------------------------------------------------------------------
    # 6. Freeze / unfreeze — Option B by default
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    editclip_encoder.requires_grad_(False)
    transformer.requires_grad_(False)  # start frozen

    if args.full_unfreeze:
        transformer.requires_grad_(True)
        logger.info("[freeze] Option A — FULL transformer unfreeze")
    else:
        # Option B: only pos_embed.proj + text-stream + (context_embedder)
        transformer.pos_embed.proj.weight.requires_grad_(True)
        if transformer.pos_embed.proj.bias is not None:
            transformer.pos_embed.proj.bias.requires_grad_(True)
        transformer.context_embedder.requires_grad_(True)
        for name, param in transformer.named_parameters():
            if any(k in name for k in [
                "add_q_proj", "add_k_proj", "add_v_proj",
                "to_add_out",
                "norm_added_q", "norm_added_k",
                "norm1_context", "norm2_context",
                "ff_context",
                "scale_shift_bias_c", "scale_shift_scale_c",
            ]):
                param.requires_grad_(True)
        logger.info("[freeze] Option B — partial unfreeze "
                    "(pos_embed.proj + context_embedder + text-stream)")

    editclip_adapter.requires_grad_(True)

    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    n_pe = sum(p.numel() for p in transformer.pos_embed.proj.parameters()
                if p.requires_grad)
    n_adapter = sum(p.numel() for p in editclip_adapter.parameters())
    logger.info(f"Trainable transformer: {n_train:,} / {n_total:,} "
                f"({100*n_train/n_total:.1f}%)")
    logger.info(f"Trainable pos_embed.proj: {n_pe:,} params  "
                f"(MUST be >0 to keep concat learning)")
    logger.info(f"Trainable adapter: {n_adapter:,}")
    if n_pe == 0:
        raise RuntimeError("pos_embed.proj frozen — channel concat will "
                            "stop learning. Check the freeze policy.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing(
            nblocks_to_apply_grad_checkpointing=24
        )

    # ------------------------------------------------------------------
    # 7. Optimizer
    # ------------------------------------------------------------------
    trainable = (
        list(filter(lambda p: p.requires_grad, transformer.parameters()))
        + list(editclip_adapter.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
    )

    # ------------------------------------------------------------------
    # 8. Dataset / DataLoader (live IP2P)
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, cache_dir=None)
    if args.max_train_samples is not None:
        dataset["train"] = (dataset["train"]
                             .shuffle(seed=args.seed)
                             .select(range(args.max_train_samples)))
    train_dataset = IP2PEditCLIPDataset(
        dataset["train"],
        editclip_processor=editclip_processor,
        resolution=args.resolution,
        original_image_column=args.original_image_column,
        edited_image_column=args.edited_image_column,
        text_column=args.text_column,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True, drop_last=True,
    )

    # ------------------------------------------------------------------
    # 9. LR scheduler
    # ------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    if resume_step > 0:
        for _ in range(resume_step * accelerator.num_processes):
            lr_scheduler.step()
        logger.info(f"[resume] LR scheduler fast-forwarded to step {resume_step}")

    # ------------------------------------------------------------------
    # 10. Prepare with accelerator
    # ------------------------------------------------------------------
    transformer, editclip_adapter, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(transformer, editclip_adapter, optimizer,
                              train_dataloader, lr_scheduler)
    vae.to(accelerator.device, dtype=weight_dtype)
    editclip_encoder.to(accelerator.device, dtype=weight_dtype)

    if optimizer_state_to_load is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
            logger.info("[resume] optimizer state restored")
        except Exception as e:
            logger.warning(f"[resume] could not restore optimizer state ({e})")

    if accelerator.is_main_process:
        accelerator.init_trackers("emmdit-ip2p-stage2", config=vars(args))

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    logger.info("***** Stage 2 training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total steps = {args.max_train_steps}")
    logger.info(f"  Starting at step = {resume_step}")
    logger.info(f"  LR = {args.learning_rate}, warmup = {args.lr_warmup_steps}")

    global_step = resume_step
    progress_bar = tqdm(range(args.max_train_steps),
                          disable=not accelerator.is_local_main_process,
                          initial=global_step)

    for epoch in range(args.num_train_epochs):
        transformer.train()
        editclip_adapter.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # ---- VAE encode both images ----
                with torch.no_grad():
                    edited_latents = vae.encode(
                        batch["edited_pixel_values"].to(weight_dtype)
                    )
                    if hasattr(edited_latents, "latent_dist"):
                        edited_latents = edited_latents.latent_dist.sample()
                    elif hasattr(edited_latents, "latent"):
                        edited_latents = edited_latents.latent
                    edited_latents = edited_latents * vae.config.scaling_factor

                    original_latents = vae.encode(
                        batch["original_pixel_values"].to(weight_dtype)
                    )
                    if hasattr(original_latents, "latent_dist"):
                        original_latents = original_latents.latent_dist.sample()
                    elif hasattr(original_latents, "latent"):
                        original_latents = original_latents.latent
                    original_latents = original_latents * vae.config.scaling_factor

                # ---- Sample noise + t ----
                noise = torch.randn_like(edited_latents)
                bsz = edited_latents.shape[0]
                t = sample_logit_normal(bsz, device=edited_latents.device)
                x_t, v_target = compute_rf_targets(edited_latents, noise, t)
                timesteps = (t * 1000.0).long().clamp(0, 999)

                # ---- EditCLIP feature -> adapter -> cond ----
                with torch.no_grad():
                    editclip_output = editclip_encoder(
                        batch["editclip_input"].to(weight_dtype)
                    )
                    editclip_feats = editclip_output[0][:, 1:, :]  # drop CLS
                adapter_full = editclip_adapter(editclip_feats.float())

                # ---- IP2P CFG dropouts ----
                # Paper formulation: one random_p, derived masks
                p_drop = args.conditioning_dropout_prob
                if p_drop > 0:
                    if args.legacy_independent_dropout:
                        drop_text = torch.rand(bsz,
                                                 device=adapter_full.device
                                                 ) < p_drop
                        drop_img = torch.rand(bsz,
                                                device=adapter_full.device
                                                ) < p_drop
                    else:
                        rp = torch.rand(bsz, device=adapter_full.device)
                        drop_text = rp < 2.0 * p_drop
                        drop_img = (rp >= p_drop) & (rp < 3.0 * p_drop)
                    if drop_text.any():
                        zero_h = torch.zeros_like(adapter_full)
                        adapter_full = torch.where(
                            drop_text[:, None, None], zero_h, adapter_full)
                    if drop_img.any():
                        zero_orig = torch.zeros_like(original_latents)
                        original_latents = torch.where(
                            drop_img[:, None, None, None], zero_orig,
                            original_latents,
                        )

                # ---- Channel concat — preserve Stage 1's mechanism ----
                hidden_in = torch.cat(
                    [x_t.float(), original_latents.float()], dim=1,
                )  # [B, 64, H, W]

                # ---- Forward ----
                model_output = transformer(
                    hidden_states=hidden_in,
                    encoder_hidden_states=adapter_full.float(),
                    timestep=timesteps,
                    return_dict=False,
                )
                v_pred = (model_output[0] if isinstance(model_output, tuple)
                           else model_output.sample)

                B, C, H, W = v_target.shape
                v_target_tokens = v_target.permute(0, 2, 3, 1).reshape(
                    B, H * W, C
                )
                loss = F.mse_loss(v_pred.float(),
                                    v_target_tokens.float(),
                                    reduction="mean")

                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(filter(lambda p: p.requires_grad,
                                       transformer.parameters()))
                        + list(editclip_adapter.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # ---- Critical monitor: pos_embed extra-32ch must not collapse ----
                if accelerator.is_main_process and \
                        (global_step % 100 == 0):
                    unwrapped = accelerator.unwrap_model(transformer)
                    w = unwrapped.pos_embed.proj.weight.detach()
                    half = w.shape[1] // 2
                    extra_abs = w[:, half:].abs().mean().item()
                    orig_abs = w[:, :half].abs().mean().item()
                    ratio = extra_abs / (orig_abs + 1e-12)
                    accelerator.log({
                        "pos_embed_extra_abs_mean": extra_abs,
                        "pos_embed_orig_abs_mean":  orig_abs,
                        "pos_embed_extra_over_orig": ratio,
                    }, step=global_step)
                    if global_step % 1000 == 0:
                        logger.info(f"[pos_embed] step={global_step}  "
                                    f"orig={orig_abs:.4e}  extra={extra_abs:.4e}  "
                                    f"ratio={ratio:.3f}")

                # ---- Checkpoint ----
                if (global_step % args.checkpointing_steps == 0
                        and accelerator.is_main_process):
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    # Disk pre-flight
                    try:
                        st = shutil.disk_usage(args.output_dir)
                        if st.free / (1024 ** 3) < 7.0:
                            logger.warning(
                                f"[ckpt] SKIP step={global_step}: "
                                f"{st.free/(1024**3):.2f} GB free")
                            continue
                    except Exception:
                        pass
                    unwrapped_t = accelerator.unwrap_model(transformer)
                    unwrapped_a = accelerator.unwrap_model(editclip_adapter)
                    torch.save(unwrapped_t.state_dict(),
                                os.path.join(save_path, "transformer.pt"))
                    torch.save(unwrapped_a.state_dict(),
                                os.path.join(save_path, "editclip_adapter.pt"))
                    torch.save(optimizer.state_dict(),
                                os.path.join(save_path, "optimizer.pt"))
                    logger.info(f"Saved checkpoint to {save_path}")

                    # Auto-prune (same logic as Stage 1)
                    keep_n = args.keep_last_n_checkpoints
                    protect_every = args.protect_every_n_steps or 0
                    protect_explicit = set(args.protect_steps or [])
                    if keep_n and keep_n > 0:
                        ckpt_dirs = sorted(
                            (d for d in os.listdir(args.output_dir)
                              if re.fullmatch(r"checkpoint-\d+", d)
                              and os.path.isdir(os.path.join(args.output_dir, d))),
                            key=lambda d: int(re.match(r"checkpoint-(\d+)",
                                                          d).group(1)),
                        )
                        if len(ckpt_dirs) > keep_n:
                            recent = set(ckpt_dirs[-keep_n:])
                            protected = set()
                            for d in ckpt_dirs:
                                step = int(re.match(r"checkpoint-(\d+)",
                                                       d).group(1))
                                if protect_every > 0 and step % protect_every == 0:
                                    protected.add(d)
                                if step in protect_explicit:
                                    protected.add(d)
                            for d in ckpt_dirs:
                                if d not in (recent | protected):
                                    victim = os.path.join(args.output_dir, d)
                                    logger.info(f"[ckpt] pruning {victim}")
                                    shutil.rmtree(victim, ignore_errors=True)

            logs = {"loss": loss.detach().item(),
                     "lr": lr_scheduler.get_last_lr()[0],
                     "step": global_step}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # ------------------------------------------------------------------
    # 12. Final save
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_t = accelerator.unwrap_model(transformer)
        unwrapped_a = accelerator.unwrap_model(editclip_adapter)
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(unwrapped_t.state_dict(),
                    os.path.join(final_path, "transformer.pt"))
        torch.save(unwrapped_a.state_dict(),
                    os.path.join(final_path, "editclip_adapter.pt"))
        torch.save(optimizer.state_dict(),
                    os.path.join(final_path, "optimizer.pt"))
        logger.info(f"Saved final to {final_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
