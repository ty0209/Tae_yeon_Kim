#!/usr/bin/env python
# coding=utf-8
"""
Quick diagnostic + visual eval for an IP2P Stage 1 checkpoint
(train_editclip_emmdit_ip2p.py output).

PRIMARY DIAGNOSTIC: source sensitivity
--------------------------------------
For each sample we run the model TWICE:

    output_full       = denoise( cat[x_t, x0_orig], text_prompt )   # IP2P
    output_no_source  = denoise( cat[x_t, zeros   ], text_prompt )  # ablated

and report

    LPIPS(output_full, output_no_source)

If this number is near 0, the channel-concat trick is NOT in use: the
model is ignoring the source latent and just doing text-to-image. If
it's > 0.15, the model is genuinely using the source.

Also run a third pass with null text:

    output_no_text    = denoise( cat[x_t, x0_orig], null_prompt )

to see source-only behaviour. The grid layout is

  [ source | output_full | output_no_source | output_no_text | target_GT ]

Standard quality metrics
------------------------
If a ground-truth target image is provided (or we're on benchmark, where
test/<id>_1.* is the GT), we also report:
  - LPIPS(output_full, target_GT)
  - SSIM(output_full, target_GT)
  - CLIP Score(output_full, edit_prompt)

Modes
-----
1) Single-sample:
   --source_image PATH  --edit_prompt "..."  [--target_image PATH]

2) Benchmark sweep (uses topbench_eval/dataset.py + task_prompts.json):
   --benchmark_root .../benchmark --task_prompts topbench_eval/task_prompts.json
   --max_samples 10

Usage
-----
python eval_emmdit_ip2p_stage1.py \
    --finetuned_ckpt /path/to/checkpoint-NNNN/transformer.pt \
    --source_image    /path/to/src.jpg \
    --edit_prompt     "turn the boy into a girl" \
    --target_image    /path/to/tgt.jpg \
    --output_dir      output/eval_ip2p_stage1_single \
    --cfg_scale       4.5 \
    --num_steps       30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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


# ---------------------------------------------------------------------------
# Helpers (copied from the trainer so this script is standalone)
# ---------------------------------------------------------------------------

def find_nitroE_weights(repo: str, filename: str,
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


def extend_pos_embed_to_concat(transformer) -> int:
    """Same extension trick the trainer applies."""
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


@torch.no_grad()
def encode_prompt(prompt: str, tokenizer, text_encoder, device, dtype,
                   max_length: int = 128):
    toks = tokenizer(
        [prompt.lower().strip()],
        padding="max_length", max_length=max_length,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    ).to(device)
    out = text_encoder(toks.input_ids, attention_mask=toks.attention_mask,
                        output_hidden_states=True)
    return out["hidden_states"][-1].to(dtype=dtype), toks.attention_mask


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


# ---------------------------------------------------------------------------
# Sampler — RF Euler, text-CFG, with channel-concat source latent
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_v(transformer, x_t, x0_src_or_zero,
               text_embeds, text_mask,
               null_embeds, null_mask,
               t_sched, cfg_scale, do_cfg):
    """Hidden = [x_t || source_or_zero] (channel concat). Optional text CFG.

    All tensors are cast to the transformer's parameter dtype so Conv2d
    bias matches input dtype. (Training uses accelerator autocast; here we
    have to be explicit.)
    """
    tdtype = next(transformer.parameters()).dtype
    batch = x_t.shape[0]
    t_in = t_sched.expand(batch).to(x_t.device)
    hin = torch.cat([x_t.to(tdtype), x0_src_or_zero.to(tdtype)], dim=1)
    if do_cfg and cfg_scale > 1.0:
        h_cat = torch.cat([hin, hin], dim=0)
        t_cat = torch.cat([t_in, t_in], dim=0)
        c_cat = torch.cat([null_embeds.to(tdtype),
                             text_embeds.to(tdtype)], dim=0)
        m_cat = torch.cat([null_mask, text_mask], dim=0)
        out = transformer(
            hidden_states=h_cat,
            encoder_hidden_states=c_cat,
            encoder_attention_mask=m_cat,
            timestep=t_cat,
            return_dict=False,
        )
        v = out[0] if isinstance(out, tuple) else out.sample
        v_un, v_co = v.chunk(2)
        return v_un + cfg_scale * (v_co - v_un)
    out = transformer(
        hidden_states=hin,
        encoder_hidden_states=text_embeds.to(tdtype),
        encoder_attention_mask=text_mask,
        timestep=t_in,
        return_dict=False,
    )
    return out[0] if isinstance(out, tuple) else out.sample


@torch.no_grad()
def sample_rf_euler(transformer, scheduler, source_or_zero, text_embeds, text_mask,
                     null_embeds, null_mask, num_steps, cfg_scale, device,
                     init_noise=None, dtype=torch.bfloat16):
    """RF Euler sampling from x_T = noise to x_0."""
    scheduler.set_timesteps(num_steps, device=device)
    timesteps_desc = scheduler.timesteps.tolist() + [0.0]

    if init_noise is None:
        # tokens_h = sample_size = source.shape[-1], channels match source
        x_t = torch.randn_like(source_or_zero).to(dtype=dtype)
    else:
        x_t = init_noise.to(dtype=dtype)

    do_cfg = cfg_scale > 1.0
    for i in tqdm(range(len(timesteps_desc) - 1),
                   desc="sample", leave=False):
        t_cur = timesteps_desc[i]
        t_next = timesteps_desc[i + 1]
        dt_norm = (float(t_next) - float(t_cur)) / 1000.0
        t_cur_t = torch.tensor([float(t_cur)], device=device)
        v = predict_v(transformer, x_t, source_or_zero,
                       text_embeds, text_mask, null_embeds, null_mask,
                       t_cur_t, cfg_scale, do_cfg)
        x_t = x_t + dt_norm * v
    return x_t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def lpips_pair(a_pil: Image.Image, b_pil: Image.Image, size: int = 256,
                net: str = "alex") -> float:
    import lpips
    if not hasattr(lpips_pair, "_model"):
        lpips_pair._model = lpips.LPIPS(net=net).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()
    model = lpips_pair._model
    dev = next(model.parameters()).device
    def _t(img):
        arr = np.asarray(img.convert("RGB").resize((size, size),
                                                      Image.LANCZOS),
                          dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0) * 2.0 - 1.0
        return t.to(dev)
    return float(model(_t(a_pil), _t(b_pil)).item())


def ssim_pair(a_pil, b_pil, size=256):
    from skimage.metrics import structural_similarity as _ssim
    def _a(img):
        return np.asarray(img.convert("RGB").resize((size, size),
                                                       Image.LANCZOS),
                           dtype=np.float32) / 255.0
    return float(_ssim(_a(a_pil), _a(b_pil), channel_axis=2, data_range=1.0))


@torch.no_grad()
def clip_score(prompt, image, model_name="openai/clip-vit-large-patch14",
                size=256):
    from transformers import CLIPModel, AutoProcessor
    if not hasattr(clip_score, "_model"):
        m = CLIPModel.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()
        for p in m.parameters():
            p.requires_grad_(False)
        clip_score._model = m
        clip_score._proc = AutoProcessor.from_pretrained(model_name)
    model = clip_score._model
    proc = clip_score._proc
    dev = next(model.parameters()).device
    img = image.convert("RGB").resize((size, size), Image.LANCZOS)
    inputs = proc(text=[prompt], images=[img], return_tensors="pt",
                   padding=True, truncation=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    out = model(input_ids=inputs["input_ids"],
                  attention_mask=inputs["attention_mask"],
                  pixel_values=inputs["pixel_values"])
    t = F.normalize(out.text_embeds, dim=-1)
    v = F.normalize(out.image_embeds, dim=-1)
    return float((t * v).sum(dim=-1).item() * 100.0)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    src_path: str
    prompt: str
    target_path: Optional[str] = None
    sample_id: str = ""


def eval_one(sample: Sample, transformer, vae, scheduler, tokenizer,
              text_encoder, device, dtype, args) -> dict:
    src_pil, src_t = load_image(sample.src_path, args.resolution)
    x0_src = encode_to_latent(src_t, vae, device, dtype)
    zeros = torch.zeros_like(x0_src)

    text_h, text_m = encode_prompt(sample.prompt, tokenizer, text_encoder,
                                     device, dtype,
                                     max_length=args.llama_max_length)
    # null embedding = the unconditional / negative cond side of CFG.
    # If --negative_prompt is set, encode it instead of "" — CFG then
    # pushes AWAY from those tokens (useful for nuking neon/saturated/vivid
    # artifacts when target prompt asks for subtle aesthetic).
    neg_text = getattr(args, "negative_prompt", "") or ""
    null_h, null_m = encode_prompt(neg_text, tokenizer, text_encoder,
                                     device, dtype,
                                     max_length=args.llama_max_length)

    # Use a SHARED noise for all three passes so differences are due to
    # conditioning, not noise.
    g = torch.Generator(device=device).manual_seed(args.seed)
    init_noise = torch.randn(*x0_src.shape, generator=g, device=device,
                              dtype=dtype)

    def _sample(src_or_zero, text_h_, text_m_):
        return sample_rf_euler(
            transformer, scheduler, src_or_zero,
            text_h_, text_m_, null_h, null_m,
            num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            device=device, init_noise=init_noise, dtype=dtype,
        )

    # 1) full: source + text
    x_full = _sample(x0_src, text_h, text_m)
    pil_full = to_pil(decode_from_latent(x_full, vae).cpu())

    # 2) source ablated: zeros + text
    x_no_src = _sample(zeros, text_h, text_m)
    pil_no_src = to_pil(decode_from_latent(x_no_src, vae).cpu())

    # 3) text ablated: source + null
    x_no_txt = _sample(x0_src, null_h, null_m)
    pil_no_txt = to_pil(decode_from_latent(x_no_txt, vae).cpu())

    # Source sensitivity = LPIPS between (full) and (no source)
    src_sens = lpips_pair(pil_full, pil_no_src, size=args.eval_size)
    # Text sensitivity (sanity)
    txt_sens = lpips_pair(pil_full, pil_no_txt, size=args.eval_size)

    out = {
        "sample_id": sample.sample_id or os.path.basename(sample.src_path),
        "prompt": sample.prompt,
        "source_sensitivity_lpips":  src_sens,
        "text_sensitivity_lpips":    txt_sens,
    }

    # Optional GT-based metrics
    target_pil = None
    if sample.target_path and os.path.isfile(sample.target_path):
        target_pil, _ = load_image(sample.target_path, args.resolution)
        out["lpips_vs_target"] = lpips_pair(pil_full, target_pil,
                                              size=args.eval_size)
        out["ssim_vs_target"]  = ssim_pair(pil_full, target_pil,
                                             size=args.eval_size)
    if sample.prompt:
        out["clip_score_x100"] = clip_score(sample.prompt, pil_full,
                                              size=args.eval_size)

    # Save grid
    panels = [src_pil, pil_full, pil_no_src, pil_no_txt]
    labels = ["source", f"full (cfg={args.cfg_scale})",
              "no_source", "no_text"]
    if target_pil is not None:
        panels.append(target_pil)
        labels.append("target_GT")
    grid = make_grid(panels, labels)
    os.makedirs(args.output_dir, exist_ok=True)
    sid = (sample.sample_id or
            os.path.splitext(os.path.basename(sample.src_path))[0])
    grid.save(os.path.join(args.output_dir, f"{sid}.png"))
    out["output_grid"] = os.path.join(args.output_dir, f"{sid}.png")
    return out


# ---------------------------------------------------------------------------
# Sample sources
# ---------------------------------------------------------------------------

def gather_single(args) -> List[Sample]:
    if not args.source_image or not args.edit_prompt:
        raise SystemExit("single-sample mode needs --source_image AND "
                          "--edit_prompt")
    return [Sample(
        src_path=args.source_image,
        prompt=args.edit_prompt,
        target_path=args.target_image,
        sample_id=os.path.splitext(os.path.basename(args.source_image))[0],
    )]


def gather_benchmark(args) -> List[Sample]:
    sys.path.insert(0, os.path.join(_HERE, "topbench_eval"))
    import dataset as bench_ds  # noqa
    tasks = bench_ds.discover_tasks(
        args.benchmark_root,
        categories=args.categories,
        task_filter=args.tasks,
    )
    with open(args.task_prompts) as f:
        raw = json.load(f)
    prompts = {k: v for k, v in raw.items() if not k.startswith("_")}

    samples: List[Sample] = []
    for t in tasks:
        key = f"{t.category}/{t.name}"
        info = prompts.get(key, {})
        prompt = (info.get("clip_prompt")
                   or info.get("target_prompt") or "")
        if not prompt:
            continue
        for pair in t.test_pairs:
            samples.append(Sample(
                src_path=pair.src_path,
                target_path=pair.ed_path,
                prompt=prompt,
                sample_id=f"{t.category}_{t.name}_{pair.pair_id}",
            ))
            if (args.max_samples_per_task and
                    sum(1 for s in samples
                         if s.sample_id.startswith(f"{t.category}_{t.name}_"))
                    >= args.max_samples_per_task):
                break
    if args.max_samples:
        samples = samples[: args.max_samples]
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    # ckpt
    p.add_argument("--finetuned_ckpt", type=str, required=True,
                   help="Path to a Stage 1 transformer.pt with 64-ch "
                        "pos_embed (output of train_editclip_emmdit_ip2p.py).")
    p.add_argument("--emmdit_repo", type=str, default="amd/Nitro-E")
    p.add_argument("--emmdit_ckpt", type=str, default="Nitro-E-512px.safetensors")
    p.add_argument("--nitroE_local_path", type=str, default=None)
    p.add_argument("--llama_model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--llama_max_length", type=int, default=128)
    p.add_argument("--negative_prompt", type=str, default="",
                   help="Optional text used in place of the empty null cond. "
                        "CFG pushes AWAY from this. Good for killing "
                        "neon/saturated/vivid artifacts.")
    # sampling
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--eval_size", type=int, default=256,
                   help="Resize images to this size before metric computation.")
    # mode 1: single sample
    p.add_argument("--source_image", type=str, default=None)
    p.add_argument("--edit_prompt", type=str, default=None)
    p.add_argument("--target_image", type=str, default=None)
    # mode 2: benchmark sweep
    p.add_argument("--benchmark_root", type=str, default=None)
    p.add_argument("--task_prompts", type=str, default=None)
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_samples_per_task", type=int, default=None)
    # output
    p.add_argument("--output_dir", type=str, default="output/eval_ip2p_stage1")
    p.add_argument("--output_csv", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # --- gather samples
    if args.benchmark_root:
        if not args.task_prompts:
            raise SystemExit("benchmark mode needs --task_prompts")
        samples = gather_benchmark(args)
    else:
        samples = gather_single(args)
    print(f"[eval] {len(samples)} samples to evaluate")

    # --- load Nitro-E base, extend to 64ch, then overlay ckpt
    print(f"[load] Nitro-E base...")
    transformer = EMMDiTTransformer(
        sample_size=args.resolution // 32,
        use_sub_attn=(args.resolution <= 512),
    )
    base_state = load_file(find_nitroE_weights(
        args.emmdit_repo, args.emmdit_ckpt, local_path=args.nitroE_local_path
    ))
    transformer.load_state_dict(base_state)
    extend_pos_embed_to_concat(transformer)

    print(f"[load] Stage 1 ckpt: {args.finetuned_ckpt}")
    t_state = torch.load(args.finetuned_ckpt, map_location="cpu")
    # Auto-pad if ckpt is 32-ch (vanilla or emmdit2 lineage).
    if "pos_embed.proj.weight" in t_state:
        ckpt_in = t_state["pos_embed.proj.weight"].shape[1]
        cur_in = transformer.pos_embed.proj.in_channels
        if ckpt_in == cur_in // 2:
            print(f"  [pad] {ckpt_in}-ch -> {cur_in}-ch (zero pad)")
            old_w = t_state["pos_embed.proj.weight"]
            pad = torch.zeros(old_w.shape[0], cur_in - ckpt_in,
                                *old_w.shape[2:], dtype=old_w.dtype)
            t_state["pos_embed.proj.weight"] = torch.cat([old_w, pad], dim=1)
    transformer.load_state_dict(t_state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    # ---- diagnostic on pos_embed.proj extra-32ch ----
    w = transformer.pos_embed.proj.weight.detach()
    half = w.shape[1] // 2
    first_abs = w[:, :half].abs().mean().item()
    extra_abs = w[:, half:].abs().mean().item()
    print(f"[diag] pos_embed.proj weight magnitude:")
    print(f"        first {half}-ch mean_abs = {first_abs:.4e}")
    print(f"        extra {half}-ch mean_abs = {extra_abs:.4e}  "
          f"(ratio {extra_abs/(first_abs+1e-12):.3f})")
    if extra_abs < first_abs * 0.01:
        print("        WARNING: extra channels are essentially zero — "
              "IP2P concat is NOT trained yet.")

    print(f"[load] Llama text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.llama_model, torch_dtype=dtype,
    ).to(device).eval()
    for pp in text_encoder.parameters():
        pp.requires_grad_(False)

    print(f"[load] DC-AE VAE...")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=dtype,
    ).to(device).eval()

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # --- iterate
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for s in tqdm(samples, desc="eval"):
        try:
            r = eval_one(s, transformer, vae, scheduler, tokenizer,
                          text_encoder, device, dtype, args)
        except Exception as e:
            print(f"[err] {s.sample_id}: {e}")
            continue
        results.append(r)

    # --- aggregate
    def _avg(key):
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print("=" * 60)
    print(f" Stage 1 diagnostic — {len(results)} samples")
    print("=" * 60)
    print(f"  source sensitivity (LPIPS full vs no_source) "
          f": {_avg('source_sensitivity_lpips'):.4f}")
    print(f"     >= 0.15 means model IS using source ✓")
    print(f"     <  0.05 means model is IGNORING source ✗")
    print(f"  text sensitivity   (LPIPS full vs no_text)   "
          f": {_avg('text_sensitivity_lpips'):.4f}")
    print(f"     (sanity — text-cond should also matter)")
    if any("lpips_vs_target" in r for r in results):
        print(f"  LPIPS vs target_GT (full): {_avg('lpips_vs_target'):.4f}")
        print(f"  SSIM  vs target_GT (full): {_avg('ssim_vs_target'):.4f}")
    if any("clip_score_x100" in r for r in results):
        print(f"  CLIP Score (full, x100) : {_avg('clip_score_x100'):.4f}")
    print("=" * 60)

    # --- save CSV
    csv_path = args.output_csv or os.path.join(
        args.output_dir, "results.csv"
    )
    if results:
        import csv
        keys = ["sample_id", "prompt",
                 "source_sensitivity_lpips", "text_sensitivity_lpips",
                 "lpips_vs_target", "ssim_vs_target", "clip_score_x100",
                 "output_grid"]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in results:
                w.writerow([r.get(k, "") for k in keys])
        print(f"[done] csv -> {csv_path}")

    # --- JSON summary
    summary = {
        "args": vars(args),
        "n_samples": len(results),
        "source_sensitivity_lpips_mean": _avg("source_sensitivity_lpips"),
        "text_sensitivity_lpips_mean":   _avg("text_sensitivity_lpips"),
        "lpips_vs_target_mean": _avg("lpips_vs_target"),
        "ssim_vs_target_mean":  _avg("ssim_vs_target"),
        "clip_score_x100_mean": _avg("clip_score_x100"),
        "pos_embed_first_abs":  first_abs,
        "pos_embed_extra_abs":  extra_abs,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary -> {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
