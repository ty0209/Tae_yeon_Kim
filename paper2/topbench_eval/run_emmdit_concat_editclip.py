"""
Run **EditCLIP + E-MMDiT (channel concat)** over the EditCLIP-style
benchmark. Uses the Stage-2 papermatch checkpoint (Linear+LN adapter,
cond per-token L2 ≈ 45) trained with IP2P-style channel concat.

Apples-to-apples with the EditCLIP paper baseline
-------------------------------------------------
- `--cfg_mode dual --cfg_text 7.0 --cfg_image 1.5` matches paper IP2P
  dual-CFG (guidance_scale=7, image_guidance_scale=1.5).
- `--cfg_mode text --cfg_scale 4.5` is simpler single-CFG, matches the
  training-time CFG dropout calibration.
- Default = dual to mirror paper's table-reported scales.

Usage
-----
python topbench_eval/run_emmdit_concat_editclip.py \
    --benchmark_root .../benchmark \
    --finetuned_ckpt .../emmdit_ip2p_stage2_papermatch/.../checkpoint-80000/transformer.pt \
    --editclip_path .../editclip_vit_l_14 \
    --output_dir .../output/topbench/emmdit_concat_editclip \
    --resolution 512 --k_exemplars 3 --num_replicates 1 \
    --cfg_mode dual --cfg_text 7.0 --cfg_image 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO]
sys.path.insert(0, _HERE)   # dataset.py
sys.path.insert(0, _REPO)   # inference_nitroE_ip2p_editclip

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
from transformers import AutoProcessor, CLIPModel

from core.models.transformer_emmdit import EMMDiTTransformer

import dataset as bench_ds  # noqa: E402
import inference_nitroE_ip2p_editclip as I  # noqa: E402  pull helpers


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


def _resolve_cond_token_mode(ckpt_path: str,
                                explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    marker = os.path.join(os.path.dirname(ckpt_path), "cond_token_mode.txt")
    if os.path.exists(marker):
        with open(marker) as f:
            return f.read().strip()
    return "patches"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Benchmark
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--k_exemplars", type=int, default=3)
    p.add_argument("--num_replicates", type=int, default=1)
    p.add_argument("--max_tests_per_task", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_output_dir", action="store_true")
    # Model
    p.add_argument("--finetuned_ckpt", type=str, required=True,
                   help="Stage-2 papermatch transformer.pt (64-ch pos_embed). "
                        "editclip_adapter.pt auto-loaded from same dir.")
    p.add_argument("--editclip_path", type=str, required=True)
    p.add_argument("--nitroE_local_path", type=str, default=None)
    p.add_argument("--dcae_local_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    # Inference
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    p.add_argument("--delta_t", type=float, default=0.01)
    p.add_argument("--cfg_mode", type=str, default="dual",
                   choices=["text", "dual"])
    p.add_argument("--cfg_scale", type=float, default=4.5,
                   help="Text CFG for cfg_mode=text.")
    p.add_argument("--cfg_text", type=float, default=7.0,
                   help="Text CFG for dual mode (paper default 7).")
    p.add_argument("--cfg_image", type=float, default=1.5,
                   help="Image CFG for dual mode (paper default 1.5).")
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--cond_token_mode", type=str, default=None,
                   choices=[None, "patches", "cls"])
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

    cond_token_mode = _resolve_cond_token_mode(args.finetuned_ckpt,
                                                  args.cond_token_mode)
    print(f"[cond] cond_token_mode = {cond_token_mode}")

    # ---- discover tasks ----
    tasks = bench_ds.discover_tasks(args.benchmark_root,
                                      categories=args.categories,
                                      task_filter=args.tasks)
    total = bench_ds.estimate_total_samples(
        tasks, num_replicates=args.num_replicates,
        max_tests_per_task=args.max_tests_per_task,
    )
    print(f"[bench] {len(tasks)} tasks, ~{total} samples "
          f"(k={args.k_exemplars}, rep={args.num_replicates})")

    # ---- load transformer + 64-ch pos_embed + ckpt ----
    print(f"[load] base Nitro-E (32 ch) -> extend to 64 ch")
    transformer = EMMDiTTransformer(
        sample_size=args.resolution // 32,
        use_sub_attn=(args.resolution <= 512),
    )
    nitroE_path = _find_nitroE("amd/Nitro-E", "Nitro-E-512px.safetensors",
                                 local_path=args.nitroE_local_path)
    transformer.load_state_dict(load_file(nitroE_path))
    I.extend_pos_embed_to_concat(transformer)

    print(f"[load] Stage-2 transformer: {args.finetuned_ckpt}")
    state = torch.load(args.finetuned_ckpt, map_location="cpu")
    pe_in = state.get("pos_embed.proj.weight", None)
    if pe_in is None or pe_in.shape[1] != 64:
        raise RuntimeError(f"Need 64-ch pos_embed ckpt, got "
                            f"{None if pe_in is None else pe_in.shape}")
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device, dtype=dtype).eval()

    # ---- adapter ----
    adapter_path = os.path.join(os.path.dirname(args.finetuned_ckpt),
                                  "editclip_adapter.pt")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"editclip_adapter.pt missing in {adapter_path}")
    adapter_state = torch.load(adapter_path, map_location="cpu")
    editclip_adapter = I.build_adapter_from_state(adapter_state)
    editclip_adapter = editclip_adapter.to(device, dtype=dtype).eval()

    # ---- VAE ----
    dcae_src = args.dcae_local_path or _find_local_dcae() or \
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    print(f"[load] DC-AE VAE: {dcae_src}")
    vae = AutoencoderDC.from_pretrained(
        dcae_src, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()

    # ---- EditCLIP encoder ----
    print(f"[load] EditCLIP encoder: {args.editclip_path}")
    ec_full = CLIPModel.from_pretrained(args.editclip_path,
                                          local_files_only=True)
    editclip_encoder = ec_full.vision_model.to(device, dtype=dtype).eval()
    editclip_processor = AutoProcessor.from_pretrained(args.editclip_path,
                                                          local_files_only=True)
    del ec_full

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # ---- iterate samples ----
    manifest: List[dict] = []
    pbar = tqdm(bench_ds.iter_samples(
                    tasks, k_exemplars=args.k_exemplars,
                    num_replicates=args.num_replicates,
                    seed=args.seed,
                    max_tests_per_task=args.max_tests_per_task),
                 total=total, desc="emmdit_concat_editclip")

    for sample in pbar:
        # ---- query -> source latent ----
        query_pil, query_t = _load_image(sample.query.src_path, args.resolution)
        x0_source = I.encode_to_latent(query_t, vae, device, dtype)

        # ---- K exemplars: encode each pair, average cond ----
        embs = []
        for ex in sample.exemplars:
            ex_in_pil, _ = _load_image(ex.src_path, args.resolution)
            ex_ed_pil, _ = _load_image(ex.ed_path, args.resolution)
            e = I.compute_edit_embeds(
                ex_in_pil, ex_ed_pil,
                editclip_encoder, editclip_processor, editclip_adapter,
                device, dtype,
                adapter_scale=args.adapter_scale,
                cond_token_mode=cond_token_mode,
            )
            embs.append(e)
        cond_embeds = torch.stack(embs, dim=0).mean(dim=0)
        null_embeds = torch.zeros_like(cond_embeds)

        # ---- random xT (IP2P style) ----
        g = torch.Generator(device=device).manual_seed(
            args.seed + sample.sample_idx
        )
        xT = torch.randn(*x0_source.shape, generator=g,
                          device=device, dtype=dtype)

        # ---- denoise ----
        x0_edit = I.denoise_from_noise(
            transformer, scheduler, x0_source,
            cond_embeds=cond_embeds, null_embeds=null_embeds,
            num_steps=args.num_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_mode=args.cfg_mode, cfg_scale=args.cfg_scale,
            cfg_text=args.cfg_text, cfg_image=args.cfg_image,
            device=device, init_noise=xT,
            desc=f"{sample.task.full_id}/{sample.query.pair_id}",
        )
        out_pixels = I.decode_from_latent(x0_edit, vae).cpu()
        out_pil = I.to_pil(out_pixels)

        # ---- save ----
        task_dir = os.path.join(args.output_dir, sample.task.category,
                                  sample.task.name)
        os.makedirs(task_dir, exist_ok=True)
        out_name = (f"{sample.query.pair_id}__rep{sample.sample_idx}"
                    f"__ex-{'-'.join(p.pair_id for p in sample.exemplars)}.png")
        out_path = os.path.join(task_dir, out_name)
        out_pil.save(out_path)

        manifest.append({
            "category": sample.task.category,
            "task": sample.task.name,
            "query_id": sample.query.pair_id,
            "query_src": sample.query.src_path,
            "query_gt_edit": sample.query.ed_path,
            "exemplars": [{"src": e.src_path, "ed": e.ed_path,
                            "id": e.pair_id} for e in sample.exemplars],
            "replicate": sample.sample_idx,
            "output": out_path,
        })

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"method": "editclip+emmdit+concat",
                    "args": vars(args), "samples": manifest}, f, indent=2)
    print(f"[done] {len(manifest)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
