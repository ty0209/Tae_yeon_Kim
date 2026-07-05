#!/usr/bin/env python
# coding=utf-8
"""
Stage 1 trainer: **Llama 3.2 + IP2P-style channel concat on E-MMDiT (Nitro-E).**

This script intentionally does NOT use EditCLIP. Its sole purpose is to
teach Nitro-E the InstructPix2Pix architectural trick — feeding the
clean source latent as an extra input channel — using Nitro-E's NATIVE
text encoder (Llama 3.2-1B) for the edit instruction. EditCLIP is
introduced in Stage 2 (separate script).

Why Llama first
~~~~~~~~~~~~~~~
Nitro-E was pretrained with Llama-3.2-1B as its text encoder. Its
text-stream (cross-attention, FFN) is already calibrated for Llama's
hidden_state distribution. Stage 1 changes ONLY the architectural input
side (32 -> 64 channels) and adapts the model to take an extra clean
latent. Conditioning stays in Llama's pre-aligned space so we don't
debug two things at once.

IP2P architectural trick (zero-init extension)
----------------------------------------------
    pos_embed.proj  before :  Conv2d(32 -> embed_dim, k=1)
    pos_embed.proj  after  :  Conv2d(64 -> embed_dim, k=1)
        weight[:, :32] = old weight   (preserves T2I prior)
        weight[:, 32:] = 0            (no info leaks at step 0)

At training step 0 the model behaves exactly like vanilla Nitro-E.
Gradients then flow into the new 32 channels and the model gradually
learns to use the original-latent side channel. Verified by sanity test:
max |vanilla_out - extended_out| = 0.0.

CFG dropout
-----------
IP2P-style independent dropouts (default 5% each):
  * text instruction -> empty prompt
  * source latent    -> zeros
These give the four CFG cases (both, text-only, image-only, both-null)
that the IP2P sampler needs.

Dataset
-------
Uses HuggingFace IP2P-style triplet datasets that provide
    original_image, edited_image, edit_prompt
columns (e.g. timbrooks/instructpix2pix-clip-filtered). Override column
names with --original_image_column / --edited_image_column /
--text_column.

Loading
-------
Nitro-E weights are searched in this priority:
  1. --nitroE_local_path
  2. ~/.cache/huggingface/hub/models--<repo>/snapshots/.../<filename>
  3. hf_hub_download (cache-aware)

No re-download on every run.

Usage
-----
accelerate launch train_editclip_emmdit_ip2p.py \
    --dataset_name /data/kimm0902_files/IP2P_datasets/data \
    --resolution 512 --train_batch_size 4 \
    --max_train_steps 50000 \
    --checkpointing_steps 2000 \
    --output_dir output/emmdit_ip2p_stage1 \
    --conditioning_dropout_prob 0.05 \
    --image_dropout_prob 0.05 \
    --gradient_checkpointing

Resume (max_train_steps is ABSOLUTE, like vanilla):
    --resume_from_checkpoint output/.../checkpoint-NNNN \
    --reuse_resume_output_dir

Filename note
-------------
The filename keeps `editclip` for historical reasons and to align with
the user's two-stage workflow naming. The script itself does NOT load
EditCLIP weights or encoder.
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

# Force ALL HF library calls fully offline. Must be set BEFORE any
# huggingface_hub / transformers / diffusers / datasets import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Nitro-E-main/ has a local 'datasets/' package that would shadow the
# pip-installed `datasets` library. Move the script's directory to the END
# of sys.path so site-packages wins for 'datasets', but local 'core' still
# imports.
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
from transformers import AutoTokenizer, AutoModelForCausalLM

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderDC
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from core.models.transformer_emmdit import EMMDiTTransformer

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Nitro-E weight discovery — cache-first, never re-download
# ---------------------------------------------------------------------------

def find_nitroE_weights(repo: str, filename: str,
                          local_path: Optional[str] = None) -> str:
    """Return a local path to Nitro-E safetensors. Priority:
       1. explicit --nitroE_local_path
       2. ~/.cache/huggingface/hub/models--<repo>/snapshots/*/<filename>
       3. fall back to hf_hub_download (uses cache anyway).

    Uses stdlib `print` rather than accelerate's logger so it can be called
    before Accelerator() is initialised (e.g. from sanity tests).
    """
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"--nitroE_local_path not found: {local_path}")
        print(f"[nitroE] using explicit local path: {local_path}")
        return local_path

    cache_root = Path(os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface")
    )) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if cache_dir.is_dir():
        snaps = cache_dir / "snapshots"
        if snaps.is_dir():
            for sn in sorted(snaps.iterdir()):
                cand = sn / filename
                if cand.is_file():
                    print(f"[nitroE] found in HF cache: {cand}")
                    return str(cand.resolve())

    print("[nitroE] not in local cache; calling hf_hub_download "
           "(will reuse cache if available)")
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
# Rectified Flow utilities
# ---------------------------------------------------------------------------

def sample_logit_normal(batch_size, m=0.0, s=1.0, device="cpu"):
    u = torch.randn(batch_size, device=device) * s + m
    return torch.sigmoid(u)


def compute_rf_targets(x0, noise, t):
    t = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t) * x0 + t * noise
    v_target = noise - x0
    return x_t, v_target


@torch.no_grad()
def encode_llama(texts: List[str], tokenizer, text_encoder, device, dtype,
                  max_length: int = 128):
    """Return (last_hidden_state, attention_mask) for a batch of strings."""
    toks = tokenizer(
        [t.lower().strip() for t in texts],
        padding="max_length", max_length=max_length, truncation=True,
        add_special_tokens=True, return_tensors="pt",
    ).to(device)
    out = text_encoder(toks.input_ids, attention_mask=toks.attention_mask,
                        output_hidden_states=True)
    return out["hidden_states"][-1].to(dtype=dtype), toks.attention_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IP2PDataset(torch.utils.data.Dataset):
    """Minimal IP2P-style triplet dataset.

    Returns:
      - original_pixel_values  [3, H, W] in [-1, 1]
      - edited_pixel_values    [3, H, W] in [-1, 1]
      - text                   str   (edit instruction)
    """
    def __init__(self, hf_dataset, resolution=512,
                  original_image_column="original_image",
                  edited_image_column="edited_image",
                  text_column="edit_prompt"):
        self.dataset = hf_dataset
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
        example = self.dataset[idx]
        orig_img = example[self.orig_col]
        edit_img = example[self.edit_col]
        orig_np = self._to_np(orig_img, self.resolution)
        edit_np = self._to_np(edit_img, self.resolution)
        # Apply same crop/flip to BOTH so they stay aligned.
        stacked = np.concatenate([orig_np, edit_np])
        stacked = torch.tensor(stacked).float()
        stacked = 2.0 * (stacked / 255.0) - 1.0
        stacked = self.train_transforms(stacked)
        original_pv, edited_pv = stacked.chunk(2)
        original_pv = original_pv.reshape(3, self.resolution, self.resolution)
        edited_pv = edited_pv.reshape(3, self.resolution, self.resolution)

        txt = example.get(self.text_col, "") or ""
        return {
            "original_pixel_values": original_pv,
            "edited_pixel_values":   edited_pv,
            "text":                  str(txt),
        }


def collate_fn(examples):
    return {
        "original_pixel_values": torch.stack(
            [e["original_pixel_values"] for e in examples]
        ).contiguous().float(),
        "edited_pixel_values": torch.stack(
            [e["edited_pixel_values"] for e in examples]
        ).contiguous().float(),
        "texts": [e["text"] for e in examples],
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Nitro-E base
    p.add_argument("--emmdit_repo", type=str, default="amd/Nitro-E")
    p.add_argument("--emmdit_ckpt", type=str, default="Nitro-E-512px.safetensors")
    p.add_argument("--nitroE_local_path", type=str, default=None,
                   help="Optional explicit path to Nitro-E safetensors. "
                        "Skips HF cache lookup if given.")
    # Llama text encoder
    p.add_argument("--llama_model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--llama_max_length", type=int, default=128)
    # Dataset
    p.add_argument("--dataset_name", type=str,
                   default="timbrooks/instructpix2pix-clip-filtered")
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
    p.add_argument("--max_train_steps", type=int, default=50000,
                   help="ABSOLUTE step limit. On resume, training stops AT "
                        "this step, NOT this many additional steps.")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    # IP2P knobs
    p.add_argument("--conditioning_dropout_prob", type=float, default=0.05,
                   help="In IP2P paper mode (default), this is the SINGLE p "
                        "from which all 3 drop cases are derived: with "
                        "prob p each -> text-only-drop / both-drop / image-"
                        "only-drop, rest (1-3p) keep both. So with p=0.05, "
                        "the unconditional 'both drop' is 5%. "
                        "In --legacy_independent_dropout mode, this is "
                        "just the text drop prob (independent of image drop).")
    p.add_argument("--image_dropout_prob", type=float, default=0.05,
                   help="Only used when --legacy_independent_dropout is set. "
                        "In IP2P paper mode this is ignored — image drop "
                        "shares the same p as text drop per the paper.")
    p.add_argument("--legacy_independent_dropout", action="store_true",
                   help="Use the OLD broken-but-simple independent dropouts "
                        "(two random draws). Default OFF — uses IP2P paper "
                        "convention (one random_p, derived masks).")
    p.add_argument("--no_channel_concat", action="store_true",
                   help="Disable the 64-ch channel-concat extension. "
                        "Degenerates to vanilla Nitro-E fine-tune. For "
                        "ablation only.")
    p.add_argument("--partial_unfreeze", action="store_true",
                   help="Train only pos_embed.proj + text-stream. By default "
                        "(unset) we train ALL transformer parameters, "
                        "matching IP2P's full-UNet fine-tune.")
    # Checkpointing
    p.add_argument("--output_dir", type=str, default="output/emmdit_ip2p_stage1")
    p.add_argument("--checkpointing_steps", type=int, default=2000)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=0,
                   help="0 = NEVER PRUNE (default, safe). Set >0 to enable "
                        "auto-prune. Pair with --protect_every_n_steps to "
                        "save milestone ckpts even when pruning is on.")
    p.add_argument("--protect_every_n_steps", type=int, default=0,
                   help="Milestone step interval that is never auto-pruned. "
                        "Only meaningful when keep_last_n_checkpoints>0.")
    p.add_argument("--protect_steps", type=int, nargs="+", default=None,
                   help="Explicit list of milestone step numbers to protect.")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--validation_steps", type=int, default=1000)
    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint-NNNN directory.")
    p.add_argument("--resume_step_override", type=int, default=None)
    p.add_argument("--reuse_resume_output_dir", action="store_true",
                   help="Continue logging into the same parent dir as the "
                        "resume checkpoint (no new timestamp subfolder).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Architecture extension: pos_embed.proj 32 -> 64 with zero-init
# ---------------------------------------------------------------------------

def extend_pos_embed_to_concat(transformer: EMMDiTTransformer) -> int:
    """Replace transformer.pos_embed.proj with a 2x-input-channel Conv2d.
    Existing weights are preserved on the first half; new half is zeroed.
    Returns the new in_channels count.
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Output dir: either reuse the resume parent, or create new timestamp.
    if args.resume_from_checkpoint and args.reuse_resume_output_dir:
        resume_dir = resolve_resume_dir(args.resume_from_checkpoint)
        args.output_dir = os.path.dirname(resume_dir)
        print(f"[resume] reusing output_dir = {args.output_dir}")
    else:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.output_dir = os.path.join(args.output_dir,
                                        f"{date_str}_lr{args.learning_rate}")
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

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

    # ------------------------------------------------------------------
    # 1. Build the transformer (vanilla 32-ch) and extend to 64-ch
    # ------------------------------------------------------------------
    weight_dtype = (torch.bfloat16 if args.mixed_precision == "bf16"
                     else torch.float16 if args.mixed_precision == "fp16"
                     else torch.float32)

    logger.info("Building E-MMDiT transformer...")
    transformer = EMMDiTTransformer(
        sample_size=args.resolution // 32,
        use_sub_attn=(args.resolution <= 512),
    )
    nitroE_path = find_nitroE_weights(args.emmdit_repo, args.emmdit_ckpt,
                                        local_path=args.nitroE_local_path)
    base_state = load_file(nitroE_path)
    transformer.load_state_dict(base_state)
    logger.info(f"Loaded Nitro-E base from {nitroE_path}")

    if not args.no_channel_concat:
        old_in = transformer.pos_embed.proj.in_channels
        new_in = extend_pos_embed_to_concat(transformer)
        logger.info(f"[ip2p] extended pos_embed.proj  {old_in} -> {new_in}  "
                    f"(extra channels zero-initialised)")

    # ------------------------------------------------------------------
    # 2. Llama text encoder + DC-AE VAE  (both FROZEN, both offline-only)
    # ------------------------------------------------------------------
    logger.info(f"Loading Llama text encoder: {args.llama_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.llama_model, torch_dtype=weight_dtype, local_files_only=True,
    ).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    logger.info("Loading DC-AE VAE...")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        torch_dtype=weight_dtype, local_files_only=True,
    ).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # ------------------------------------------------------------------
    # 3. Resume: load saved transformer state if any
    # ------------------------------------------------------------------
    resume_step = 0
    resume_dir = None
    optimizer_state_to_load = None
    if args.resume_from_checkpoint:
        resume_dir = resolve_resume_dir(args.resume_from_checkpoint)
        resume_step = args.resume_step_override or parse_resume_step(resume_dir)
        logger.info(f"[resume] loading from {resume_dir} (step {resume_step})")

        t_path = os.path.join(resume_dir, "transformer.pt")
        o_path = os.path.join(resume_dir, "optimizer.pt")
        if not os.path.exists(t_path):
            raise FileNotFoundError(f"transformer.pt not found in {resume_dir}")

        t_state = torch.load(t_path, map_location="cpu")

        # Auto-pad pos_embed.proj.weight if resuming a 32-ch ckpt into a
        # 64-ch model (e.g. resuming from a vanilla-Nitro-E checkpoint).
        if (not args.no_channel_concat
                and "pos_embed.proj.weight" in t_state):
            ckpt_in = t_state["pos_embed.proj.weight"].shape[1]
            cur_in = transformer.pos_embed.proj.in_channels
            if ckpt_in == cur_in // 2:
                logger.info(f"[resume] ckpt has {ckpt_in}-ch pos_embed; "
                            f"zero-padding to {cur_in}-ch")
                old_w = t_state["pos_embed.proj.weight"]
                pad = torch.zeros(old_w.shape[0], cur_in - ckpt_in,
                                    *old_w.shape[2:], dtype=old_w.dtype)
                t_state["pos_embed.proj.weight"] = torch.cat([old_w, pad], dim=1)
            elif ckpt_in != cur_in:
                raise RuntimeError(
                    f"[resume] pos_embed.proj channel mismatch: "
                    f"ckpt={ckpt_in}, model={cur_in}"
                )
        transformer.load_state_dict(t_state, strict=True)
        logger.info("[resume] transformer state restored")

        if os.path.exists(o_path):
            optimizer_state_to_load = torch.load(o_path, map_location="cpu")
            logger.info("[resume] optimizer.pt found — will restore moments")
        else:
            logger.info("[resume] no optimizer.pt found — fresh Adam moments")

    # ------------------------------------------------------------------
    # 4. Freeze / unfreeze
    # ------------------------------------------------------------------
    # Default: train EVERY transformer parameter (matches IP2P full UNet
    # fine-tune). With --partial_unfreeze we restrict to the layers that
    # MUST learn (pos_embed.proj + text-stream).
    transformer.requires_grad_(True)
    if args.partial_unfreeze:
        transformer.requires_grad_(False)
        # ALWAYS unfreeze the patch embed (the new 32 channels need to learn).
        transformer.pos_embed.proj.weight.requires_grad_(True)
        if transformer.pos_embed.proj.bias is not None:
            transformer.pos_embed.proj.bias.requires_grad_(True)
        # context_embedder + text-stream cross-attn layers (Llama cond path)
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

    # Sanity counts
    trainable_params = sum(p.numel() for p in transformer.parameters()
                            if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Trainable transformer params: {trainable_params:,} / "
                f"{total_params:,} ({100*trainable_params/total_params:.1f}%)")
    pe_train = sum(p.numel() for p in transformer.pos_embed.proj.parameters()
                    if p.requires_grad)
    logger.info(f"Trainable pos_embed.proj:    {pe_train:,} params "
                f"(MUST be >0 for IP2P concat to learn)")
    if pe_train == 0:
        raise RuntimeError("pos_embed.proj is frozen — IP2P channel concat "
                            "cannot learn. Check freeze logic.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing(
            nblocks_to_apply_grad_checkpointing=24
        )

    # ------------------------------------------------------------------
    # 5. Optimizer
    # ------------------------------------------------------------------
    trainable = list(filter(lambda p: p.requires_grad,
                              transformer.parameters()))
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
    )

    # ------------------------------------------------------------------
    # 6. Dataset / DataLoader
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, cache_dir=None)
    if args.max_train_samples is not None:
        dataset["train"] = (dataset["train"]
                             .shuffle(seed=args.seed)
                             .select(range(args.max_train_samples)))
    train_dataset = IP2PDataset(
        dataset["train"],
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
    # 7. LR scheduler
    # ------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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
        logger.info(f"[resume] LR scheduler fast-forwarded to step "
                    f"{resume_step}")

    # ------------------------------------------------------------------
    # 8. Prepare with accelerator
    # ------------------------------------------------------------------
    transformer, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(transformer, optimizer, train_dataloader,
                              lr_scheduler)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if optimizer_state_to_load is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
            logger.info("[resume] optimizer state restored")
        except Exception as e:
            logger.warning(f"[resume] could not restore optimizer state ({e}). "
                           f"Continuing with fresh Adam moments.")

    if accelerator.is_main_process:
        accelerator.init_trackers("emmdit-ip2p-stage1", config=vars(args))

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Starting from step = {resume_step}")

    global_step = resume_step
    progress_bar = tqdm(range(args.max_train_steps),
                         disable=not accelerator.is_local_main_process,
                         initial=global_step)

    for epoch in range(args.num_train_epochs):
        transformer.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    # VAE encode both images.
                    edited_latents = vae.encode(
                        batch["edited_pixel_values"].to(weight_dtype)
                    )
                    if hasattr(edited_latents, 'latent_dist'):
                        edited_latents = edited_latents.latent_dist.sample()
                    elif hasattr(edited_latents, 'latent'):
                        edited_latents = edited_latents.latent
                    edited_latents = edited_latents * vae.config.scaling_factor

                    if not args.no_channel_concat:
                        original_latents = vae.encode(
                            batch["original_pixel_values"].to(weight_dtype)
                        )
                        if hasattr(original_latents, 'latent_dist'):
                            original_latents = original_latents.latent_dist.sample()
                        elif hasattr(original_latents, 'latent'):
                            original_latents = original_latents.latent
                        original_latents = original_latents * vae.config.scaling_factor

                # Sample noise + timestep for RF.
                noise = torch.randn_like(edited_latents)
                bsz = edited_latents.shape[0]
                t = sample_logit_normal(bsz, device=edited_latents.device)
                x_t, v_target = compute_rf_targets(edited_latents, noise, t)
                timesteps = (t * 1000.0).long().clamp(0, 999)

                # ---- Llama text embeddings ----
                texts = batch["texts"]
                llama_hidden, llama_mask = encode_llama(
                    texts, tokenizer, text_encoder,
                    accelerator.device, weight_dtype,
                    max_length=args.llama_max_length,
                )

                # ---- IP2P CFG dropouts ----
                # IP2P paper formulation (default): one random_p drives the
                # three drop cases — `text-only / both / image-only` each
                # at probability p. The unconditional 'both-drop' case is
                # essential for CFG sampling baseline.
                # Reference: https://github.com/timothybrooks/instruct-pix2pix
                #
                #   p = conditioning_dropout_prob
                #   random_p ~ U[0,1)
                #     [0,   p):  text drop only
                #     [p,   2p): BOTH drop          <- unconditional baseline
                #     [2p,  3p): image drop only
                #     [3p,  1):  both kept
                p_drop = args.conditioning_dropout_prob
                if p_drop > 0:
                    if args.legacy_independent_dropout:
                        # Old behaviour: two independent rolls.
                        drop_text = torch.rand(
                            bsz, device=llama_hidden.device
                        ) < args.conditioning_dropout_prob
                        drop_img = torch.rand(
                            bsz, device=llama_hidden.device
                        ) < args.image_dropout_prob
                    else:
                        # IP2P paper: derive both masks from one random_p.
                        rp = torch.rand(bsz, device=llama_hidden.device)
                        drop_text = rp < 2.0 * p_drop          # [0, 2p)
                        drop_img  = (rp >= p_drop) & (rp < 3.0 * p_drop)  # [p, 3p)

                    # Apply text drop
                    if drop_text.any():
                        zero_h = torch.zeros_like(llama_hidden)
                        zero_m = torch.zeros_like(llama_mask)
                        llama_hidden = torch.where(
                            drop_text[:, None, None], zero_h, llama_hidden)
                        llama_mask = torch.where(
                            drop_text[:, None], zero_m, llama_mask)

                    # Apply image drop (zero out source latent)
                    if not args.no_channel_concat and drop_img.any():
                        zero_orig = torch.zeros_like(original_latents)
                        m = drop_img[:, None, None, None]
                        original_latents = torch.where(
                            m, zero_orig, original_latents)

                # ---- Build the 64-ch input (or 32-ch if concat disabled) ----
                if not args.no_channel_concat:
                    hidden_in = torch.cat(
                        [x_t.float(), original_latents.float()],
                        dim=1,
                    )  # [B, 64, H, W]
                else:
                    hidden_in = x_t.float()

                # ---- Forward ----
                model_output = transformer(
                    hidden_states=hidden_in,
                    encoder_hidden_states=llama_hidden.float(),
                    encoder_attention_mask=llama_mask,
                    timestep=timesteps,
                    return_dict=False,
                )
                v_pred = (model_output[0]
                            if isinstance(model_output, tuple)
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
                                       transformer.parameters())),
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

                # Periodic diagnostic: how much have the NEW extra-32ch
                # weights moved off zero? Should grow over time.
                if accelerator.is_main_process and \
                        (global_step % 100 == 0) and \
                        (not args.no_channel_concat):
                    unwrapped = accelerator.unwrap_model(transformer)
                    w = unwrapped.pos_embed.proj.weight.detach()
                    half = w.shape[1] // 2
                    extra_abs = w[:, half:].abs().mean().item()
                    orig_abs = w[:, :half].abs().mean().item()
                    accelerator.log({
                        "pos_embed_extra_abs_mean": extra_abs,
                        "pos_embed_orig_abs_mean":  orig_abs,
                        "pos_embed_extra_over_orig": (
                            extra_abs / (orig_abs + 1e-12)
                        ),
                    }, step=global_step)

                # Checkpoint
                if (global_step % args.checkpointing_steps == 0
                        and accelerator.is_main_process):
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    # Pre-flight disk check (~7GB safety margin).
                    try:
                        st = shutil.disk_usage(args.output_dir)
                        free_gb = st.free / (1024 ** 3)
                        if free_gb < 7.0:
                            logger.warning(
                                f"[ckpt] SKIP step={global_step}: only "
                                f"{free_gb:.2f} GB free under "
                                f"{args.output_dir}"
                            )
                            continue
                    except Exception:
                        pass
                    unwrapped_t = accelerator.unwrap_model(transformer)
                    torch.save(unwrapped_t.state_dict(),
                                os.path.join(save_path, "transformer.pt"))
                    torch.save(optimizer.state_dict(),
                                os.path.join(save_path, "optimizer.pt"))
                    logger.info(f"Saved checkpoint to {save_path}")

                    # Auto-prune
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
                            keep_set = recent | protected
                            for d in ckpt_dirs:
                                if d not in keep_set:
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
    # 10. Save final
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_t = accelerator.unwrap_model(transformer)
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(unwrapped_t.state_dict(),
                    os.path.join(final_path, "transformer.pt"))
        torch.save(optimizer.state_dict(),
                    os.path.join(final_path, "optimizer.pt"))
        logger.info(f"Saved final model to {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
