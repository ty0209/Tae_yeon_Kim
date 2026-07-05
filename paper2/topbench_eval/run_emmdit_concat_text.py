"""
Run **Llama-3.2 + E-MMDiT (channel concat)** over the EditCLIP-style
benchmark. Uses the Stage-1 IP2P ckpt (Llama text encoder, 64-ch
pos_embed channel concat). No PnP / no inversion — random xT, IP2P style.

This is the text-conditioned counterpart of run_emmdit_concat_editclip.py.
Same backbone, same channel-concat mechanism, different encoder. The
fair-comparison companion for measuring "what does swapping Llama-text
for EditCLIP-visual buy us on TOP benchmark?".

Text instructions come from task_prompts.json (per-task target_prompt).
This method does NOT use exemplars — only the test query image is edited
against the task's target text.

Manifest entries include `target_prompt` and `clip_prompt` so compare.py
can run CLIP Score against the exact same per-task text used here.

Usage
-----
python topbench_eval/run_emmdit_concat_text.py \
    --benchmark_root .../benchmark \
    --finetuned_ckpt .../emmdit_ip2p_stage1/.../checkpoint-100000/transformer.pt \
    --task_prompts topbench_eval/task_prompts.json \
    --output_dir .../output/topbench/emmdit_concat_text \
    --resolution 512 --num_steps 30 --cfg_scale 4.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO]
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.transformer_emmdit import EMMDiTTransformer

import dataset as bench_ds  # noqa: E402
# Reuse helpers from Stage 1 single-image eval — keeps behavior identical.
import eval_emmdit_ip2p_stage1 as S1  # noqa: E402


def _find_local_dcae() -> Optional[str]:
    repo = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    cache_root = Path(os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    snaps = cache_dir / "snapshots"
    if snaps.is_dir():
        for sn in sorted(snaps.iterdir()):
            if (sn / "config.json").exists():
                return str(sn.resolve())
    return None


def _find_nitroE(repo: str, filename: str,
                   local_path: Optional[str] = None) -> str:
    if local_path:
        return local_path
    cache_root = Path(os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    snaps = cache_dir / "snapshots"
    if snaps.is_dir():
        for sn in sorted(snaps.iterdir()):
            cand = sn / filename
            if cand.is_file():
                return str(cand.resolve())
    return hf_hub_download(repo, filename)


def _load_image(path: str, resolution: int):
    img = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS
    )
    return img, transforms.ToTensor()(img).unsqueeze(0)


def _denoise_text(transformer, scheduler, x0_source,
                    text_h, text_m, null_h, null_m,
                    num_steps: int, cfg_scale: float,
                    device, init_noise):
    """Mirror of eval_emmdit_ip2p_stage1's sample_rf_euler with channel
    concat. Single-CFG over text cond. Source always in channel input.
    """
    return S1.sample_rf_euler(
        transformer, scheduler, x0_source,
        text_h, text_m, null_h, null_m,
        num_steps=num_steps, cfg_scale=cfg_scale,
        device=device, init_noise=init_noise, dtype=init_noise.dtype,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Benchmark
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--max_tests_per_task", type=int, default=None)
    p.add_argument("--k_exemplars_for_metric", type=int, default=3,
                   help="Exemplars SAMPLED FROM SAME TASK and stored in "
                        "manifest for EC2EC metric computation only. The "
                        "text method does NOT use them for generation. "
                        "This keeps the metric apples-to-apples with the "
                        "EditCLIP method.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_output_dir", action="store_true")
    p.add_argument("--task_prompts", type=str, required=True,
                   help="JSON keyed by '<category>/<task>' with "
                        "source_prompt/target_prompt/clip_prompt.")
    # Model
    p.add_argument("--finetuned_ckpt", type=str, required=True,
                   help="Stage 1 IP2P transformer.pt (64-ch pos_embed).")
    p.add_argument("--nitroE_local_path", type=str, default=None)
    p.add_argument("--dcae_local_path", type=str, default=None)
    p.add_argument("--llama_model", type=str,
                   default="meta-llama/Llama-3.2-1B")
    p.add_argument("--llama_max_length", type=int, default=128)
    p.add_argument("--output_dir", type=str, required=True)
    # Inference
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    if args.clean_output_dir and os.path.isdir(args.output_dir):
        print(f"[clean] removing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

    # ---- task prompts ----
    with open(args.task_prompts) as f:
        raw_prompts = json.load(f)
    task_prompts = {k: v for k, v in raw_prompts.items()
                     if not k.startswith("_")}
    print(f"[prompts] loaded {len(task_prompts)} task prompts")

    # ---- discover tasks (text method: no exemplars; k=1 dummy) ----
    tasks = bench_ds.discover_tasks(args.benchmark_root,
                                      categories=args.categories,
                                      task_filter=args.tasks)
    # Filter tasks to those that have prompts.
    tasks = [t for t in tasks if f"{t.category}/{t.name}" in task_prompts]
    print(f"[bench] {len(tasks)} tasks with prompts available")

    total = bench_ds.estimate_total_samples(
        tasks, num_replicates=1, max_tests_per_task=args.max_tests_per_task,
    )

    # ---- load transformer + 64-ch pos_embed + Stage 1 ckpt ----
    print(f"[load] base Nitro-E (32 ch) -> extend to 64 ch")
    transformer = EMMDiTTransformer(
        sample_size=args.resolution // 32,
        use_sub_attn=(args.resolution <= 512),
    )
    nitroE_path = _find_nitroE("amd/Nitro-E", "Nitro-E-512px.safetensors",
                                 local_path=args.nitroE_local_path)
    transformer.load_state_dict(load_file(nitroE_path))
    S1.extend_pos_embed_to_concat(transformer)

    print(f"[load] Stage-1 transformer: {args.finetuned_ckpt}")
    state = torch.load(args.finetuned_ckpt, map_location="cpu")
    pe_in = state.get("pos_embed.proj.weight", None)
    if pe_in is None or pe_in.shape[1] != 64:
        raise RuntimeError(f"Need 64-ch pos_embed Stage 1 ckpt, got "
                            f"{None if pe_in is None else pe_in.shape}")
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    # ---- VAE ----
    dcae_src = args.dcae_local_path or _find_local_dcae() or \
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    print(f"[load] DC-AE VAE: {dcae_src}")
    vae = AutoencoderDC.from_pretrained(
        dcae_src, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()

    # ---- Llama text encoder ----
    print(f"[load] Llama tokenizer + encoder: {args.llama_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model,
                                                 local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.llama_model, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()
    for pp in text_encoder.parameters():
        pp.requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # ---- null embeddings (once) ----
    null_h, null_m = S1.encode_prompt(
        "", tokenizer, text_encoder, device, dtype,
        max_length=args.llama_max_length,
    )

    # ---- iterate samples ----
    # NOTE: k_exemplars set here for METRIC purposes (EC2EC needs an
    # exemplar pair). The text inference itself ignores them.
    manifest: List[dict] = []
    pbar = tqdm(bench_ds.iter_samples(
                    tasks,
                    k_exemplars=args.k_exemplars_for_metric,
                    num_replicates=1,
                    seed=args.seed,
                    max_tests_per_task=args.max_tests_per_task),
                 total=total, desc="emmdit_concat_text")

    for sample in pbar:
        prompt_key = f"{sample.task.category}/{sample.task.name}"
        info = task_prompts.get(prompt_key, {})
        target_prompt = info.get("target_prompt", "")
        clip_prompt = info.get("clip_prompt") or target_prompt
        if not target_prompt:
            continue  # safety

        # ---- query -> source latent ----
        query_pil, query_t = _load_image(sample.query.src_path, args.resolution)
        x0_source = S1.encode_to_latent(query_t, vae, device, dtype)

        # ---- text encoding (target prompt) ----
        text_h, text_m = S1.encode_prompt(
            target_prompt, tokenizer, text_encoder, device, dtype,
            max_length=args.llama_max_length,
        )

        # ---- random xT (IP2P style) ----
        g = torch.Generator(device=device).manual_seed(args.seed)
        xT = torch.randn(*x0_source.shape, generator=g,
                          device=device, dtype=dtype)

        # ---- denoise ----
        x0_edit = _denoise_text(
            transformer, scheduler, x0_source,
            text_h, text_m, null_h, null_m,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            device=device, init_noise=xT,
        )
        out_pixels = S1.decode_from_latent(x0_edit, vae).cpu()
        out_pil = S1.to_pil(out_pixels)

        # ---- save ----
        task_dir = os.path.join(args.output_dir, sample.task.category,
                                  sample.task.name)
        os.makedirs(task_dir, exist_ok=True)
        out_name = f"{sample.query.pair_id}__rep0.png"
        out_path = os.path.join(task_dir, out_name)
        out_pil.save(out_path)

        manifest.append({
            "category": sample.task.category,
            "task": sample.task.name,
            "query_id": sample.query.pair_id,
            "query_src": sample.query.src_path,
            "query_gt_edit": sample.query.ed_path,
            # Exemplars: NOT used for inference, but stored so compare.py
            # can compute EC2EC against the same exemplar pool the
            # EditCLIP method draws from. Apples-to-apples.
            "exemplars": [{"src": e.src_path, "ed": e.ed_path,
                            "id": e.pair_id} for e in sample.exemplars],
            "replicate": 0,
            "output": out_path,
            "target_prompt": target_prompt,
            "clip_prompt": clip_prompt,
        })

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"method": "llama+emmdit+concat",
                    "args": vars(args), "samples": manifest}, f, indent=2)
    print(f"[done] {len(manifest)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
