"""
Head-to-head profiling of two RF-Solver+PnP inference paths on the SAME
inputs:

  A. inference_nitroE_rfsolver5.py        (EditCLIP + Nitro-E + PnP)
  B. inference_nitroE_rfsolver_text_pnp.py (Llama-3.2 + Nitro-E + PnP)

Reports per method:
  * parameter count of every model component (text/visual encoder, adapter,
    transformer, VAE) and the total
  * peak GPU memory used during a single end-to-end inference
    (image load -> encode cond -> invert -> source-record -> edit -> decode)
  * wall-clock time per sample, averaged over N runs after K warmups

Each method is loaded, profiled, and then UNLOADED before the next, so the
VRAM measurements aren't polluted by the other pipeline sitting alongside.

Usage
-----
python topbench_eval/benchmark_compare.py \
    --benchmark_root .../benchmark \
    --finetuned_ckpt .../checkpoint-NNNN/transformer.pt \
    --editclip_path  .../editclip_omnistyle_resumed/.../checkpoint-100000 \
    --task_prompts   topbench_eval/task_prompts.json \
    --output_json    .../output/benchmark_compare.json \
    --warmup 2 --measure 5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as bench_ds  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_params(n: int) -> str:
    """1234567 -> '1.23M' / '1234567890' -> '1.23B'."""
    for unit, scale in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)


def fmt_gb(bytes_: int) -> str:
    return f"{bytes_ / (1024 ** 3):.2f} GB"


def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def free_modules(*mods):
    for m in mods:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def count_params(mod) -> int:
    return sum(p.numel() for p in mod.parameters()) if mod is not None else 0


def pick_inputs(benchmark_root: str, n: int) -> List[Tuple[bench_ds.Pair, bench_ds.Pair]]:
    """Pick (query, exemplar) pairs from the FIRST discovered task. We want
    consistent inputs so the two methods are compared on identical work.
    """
    tasks = bench_ds.discover_tasks(benchmark_root)
    if not tasks:
        raise RuntimeError("no tasks found under " + benchmark_root)
    t = tasks[0]
    out = []
    # cycle query and exemplar from test/train pools
    for i in range(n):
        q = t.test_pairs[i % len(t.test_pairs)]
        e = t.train_pairs[i % len(t.train_pairs)]
        out.append((q, e, t))
    return out


# ---------------------------------------------------------------------------
# Method A: EditCLIP + Nitro-E + PnP  (inference_nitroE_rfsolver5)
# ---------------------------------------------------------------------------

def profile_editclip(args, samples) -> dict:
    print("\n" + "=" * 70)
    print("[A] EditCLIP + Nitro-E + PnP")
    print("=" * 70)
    reset_cuda_peak()
    base_alloc = cuda_peak_bytes()

    import inference_nitroE_rfsolver5 as rfs5  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

    print(f"[A] loading models...")
    load_start = time.perf_counter()
    (transformer, adapter, vae, editclip_encoder, editclip_processor,
     scheduler) = rfs5.load_models(
        args.finetuned_ckpt, args.editclip_path,
        args.resolution, device, dtype,
    )
    load_time = time.perf_counter() - load_start

    # ----- param counts -----
    params = {
        "editclip_encoder": count_params(editclip_encoder),
        "editclip_adapter": count_params(adapter),
        "transformer":      count_params(transformer),
        "vae":              count_params(vae),
    }
    params["total"] = sum(params.values())
    print("[A] params:")
    for k, v in params.items():
        print(f"     {k:20s} {fmt_params(v):>10s}  ({v:,})")

    # ----- PnP controller (defaults match the runner) -----
    pnp_stop_step = args.num_denoising_steps
    pnp_controller = rfs5.PnPController(
        layer_start=args.pnp_layer_start,
        layer_end=args.pnp_layer_end,
        step_start=0, step_end=pnp_stop_step,
        inject_q=True, inject_k=True, verbose=False,
    )
    pnp_controller.install_hooks(transformer)

    # ----- one end-to-end sample (closure) -----
    @torch.no_grad()
    def one_sample(q_pair, ex_pair):
        ex_in = Image.open(ex_pair.src_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS)
        ex_ed = Image.open(ex_pair.ed_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS)
        edit_embeds = rfs5.compute_edit_embeds(
            ex_in, ex_ed, editclip_encoder, editclip_processor, adapter,
            device, dtype, adapter_scale=args.adapter_scale,
        )
        null_embeds = torch.zeros_like(edit_embeds)

        query_pil = Image.open(q_pair.src_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS)
        query_tensor = transforms.ToTensor()(query_pil).unsqueeze(0)
        x0 = rfs5.encode_to_latent(query_tensor, vae, device, dtype)

        xT = rfs5.rfsolver_inversion(
            transformer, scheduler, x0,
            null_embeds=null_embeds, dummy_embeds=null_embeds,
            num_steps=args.num_inversion_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            device=device,
        )
        pnp_controller.cache.clear()
        pnp_controller.set_mode("record")
        pnp_controller.reset_counters()
        rfs5.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=null_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_scale=1.0, device=device,
            pnp_controller=pnp_controller, desc="A.src-rec",
        )
        pnp_controller.set_mode("inject")
        x0_edit = rfs5.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=edit_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_scale=args.cfg_scale, device=device,
            pnp_controller=pnp_controller, desc="A.edit",
        )
        pnp_controller.set_mode("off")
        _ = rfs5.decode_from_latent(x0_edit, vae).cpu()

    # ----- warmup + measure -----
    print(f"[A] warmup x{args.warmup}, measure x{args.measure}")
    for q, e, _t in samples[: args.warmup]:
        one_sample(q, e)
    reset_cuda_peak()

    times = []
    for q, e, _t in samples[args.warmup:args.warmup + args.measure]:
        t0 = time.perf_counter()
        one_sample(q, e)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    peak = cuda_peak_bytes()

    pnp_controller.remove_hooks()

    result = {
        "method": "editclip+nitroE+pnp",
        "params": params,
        "load_time_s": load_time,
        "times_s": times,
        "time_mean_s": float(sum(times) / len(times)),
        "time_min_s":  float(min(times)),
        "time_max_s":  float(max(times)),
        "peak_vram_bytes": peak,
        "peak_vram_gb": peak / (1024 ** 3),
    }
    print(f"[A] mean time/sample: {result['time_mean_s']:.2f}s  "
          f"(min {result['time_min_s']:.2f}, max {result['time_max_s']:.2f})")
    print(f"[A] peak VRAM during measure window: {fmt_gb(peak)}")

    # cleanup before next method
    free_modules(transformer, adapter, vae, editclip_encoder, scheduler,
                  pnp_controller)
    return result


# ---------------------------------------------------------------------------
# Method B: Llama-3.2 + Nitro-E + PnP  (inference_nitroE_rfsolver_text_pnp)
# ---------------------------------------------------------------------------

def profile_llama(args, samples, task_prompts: dict) -> dict:
    print("\n" + "=" * 70)
    print("[B] Llama-3.2 + Nitro-E + PnP")
    print("=" * 70)
    reset_cuda_peak()

    import inference_nitroE_rfsolver_text_pnp as rfst_pnp  # noqa: E402
    from inference_nitroE_rfsolver_text import (
        load_models as load_text_models,
        encode_prompt, load_image, encode_to_latent, decode_from_latent,
    )
    from inference_nitroE_rfsolver5 import PnPController

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

    print(f"[B] loading models...")
    load_start = time.perf_counter()
    tokenizer, text_encoder, vae, transformer, scheduler = load_text_models(
        args.repo_name, args.ckpt_name, args.resolution, device, dtype,
    )
    load_time = time.perf_counter() - load_start

    params = {
        "llama_text_encoder": count_params(text_encoder),
        "transformer":        count_params(transformer),
        "vae":                count_params(vae),
    }
    params["total"] = sum(params.values())
    print("[B] params:")
    for k, v in params.items():
        print(f"     {k:20s} {fmt_params(v):>10s}  ({v:,})")

    pnp_stop_step = args.num_denoising_steps
    pnp_controller = PnPController(
        layer_start=args.pnp_layer_start,
        layer_end=args.pnp_layer_end,
        step_start=0, step_end=pnp_stop_step,
        inject_q=True, inject_k=True, verbose=False,
    )
    pnp_controller.install_hooks(transformer)

    nul_emb, nul_mask = encode_prompt("", tokenizer, text_encoder, device,
                                       dtype, max_length=args.llama_max_length)

    @torch.no_grad()
    def one_sample(q_pair, task):
        key = f"{task.category}/{task.name}"
        info = task_prompts.get(key, {})
        tgt_text = info.get("target_prompt", "")
        src_text = info.get("source_prompt", "")
        if src_text:
            src_emb, src_mask = encode_prompt(src_text, tokenizer, text_encoder,
                                                device, dtype,
                                                max_length=args.llama_max_length)
        else:
            src_emb, src_mask = nul_emb, nul_mask
        tgt_emb, tgt_mask = encode_prompt(tgt_text, tokenizer, text_encoder,
                                            device, dtype,
                                            max_length=args.llama_max_length)

        _q_pil, query_tensor = load_image(q_pair.src_path, args.resolution)
        x0 = encode_to_latent(query_tensor, vae, device, dtype)

        xT = rfst_pnp.rfsolver_inversion(
            transformer, scheduler, x0,
            cond_embeds=src_emb, cond_mask=src_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_inversion_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_scale=1.0, device=device, desc="B.inv",
        )
        pnp_controller.cache.clear()
        pnp_controller.set_mode("record")
        pnp_controller.reset_counters()
        rfst_pnp.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=nul_emb, cond_mask=nul_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_scale=1.0, device=device,
            pnp_controller=pnp_controller, desc="B.src-rec",
        )
        pnp_controller.set_mode("inject")
        x0_edit = rfst_pnp.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=tgt_emb, cond_mask=tgt_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t, order=args.solver_order,
            cfg_scale=args.cfg_scale, device=device,
            pnp_controller=pnp_controller, desc="B.edit",
        )
        pnp_controller.set_mode("off")
        _ = decode_from_latent(x0_edit, vae).cpu()

    print(f"[B] warmup x{args.warmup}, measure x{args.measure}")
    for q, _e, t in samples[: args.warmup]:
        one_sample(q, t)
    reset_cuda_peak()

    times = []
    for q, _e, t in samples[args.warmup:args.warmup + args.measure]:
        t0 = time.perf_counter()
        one_sample(q, t)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    peak = cuda_peak_bytes()

    pnp_controller.remove_hooks()

    result = {
        "method": "llama+nitroE+pnp",
        "params": params,
        "load_time_s": load_time,
        "times_s": times,
        "time_mean_s": float(sum(times) / len(times)),
        "time_min_s":  float(min(times)),
        "time_max_s":  float(max(times)),
        "peak_vram_bytes": peak,
        "peak_vram_gb": peak / (1024 ** 3),
    }
    print(f"[B] mean time/sample: {result['time_mean_s']:.2f}s  "
          f"(min {result['time_min_s']:.2f}, max {result['time_max_s']:.2f})")
    print(f"[B] peak VRAM during measure window: {fmt_gb(peak)}")

    free_modules(text_encoder, transformer, vae, scheduler, tokenizer,
                  pnp_controller)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_comparison(a: dict, b: dict):
    print("\n" + "=" * 70)
    print("Head-to-head comparison")
    print("=" * 70)

    fields = [
        ("total params",   lambda x: fmt_params(x["params"]["total"]),
                            lambda x: x["params"]["total"]),
        ("peak VRAM",       lambda x: fmt_gb(x["peak_vram_bytes"]),
                            lambda x: x["peak_vram_bytes"]),
        ("time / sample",   lambda x: f"{x['time_mean_s']:.2f}s",
                            lambda x: x['time_mean_s']),
        ("load time",       lambda x: f"{x['load_time_s']:.1f}s",
                            lambda x: x['load_time_s']),
    ]
    print(f"{'metric':<20s} {'EditCLIP+Nitro-E':>20s} "
          f"{'Llama+Nitro-E':>20s} {'B/A ratio':>12s}")
    print("-" * 74)
    for label, fmt, num in fields:
        a_v = num(a)
        b_v = num(b)
        ratio = b_v / a_v if a_v else float("nan")
        print(f"{label:<20s} {fmt(a):>20s} {fmt(b):>20s} {ratio:>12.2f}x")

    print("\nPer-component params:")
    print(f"{'component':<22s} {'A (EditCLIP)':>15s} {'B (Llama)':>15s}")
    print("-" * 56)
    all_keys = sorted(set(a["params"].keys()) | set(b["params"].keys()))
    for k in all_keys:
        if k == "total":
            continue
        va = a["params"].get(k, 0)
        vb = b["params"].get(k, 0)
        print(f"{k:<22s} {fmt_params(va):>15s} {fmt_params(vb):>15s}")
    print(f"{'total':<22s} {fmt_params(a['params']['total']):>15s} "
          f"{fmt_params(b['params']['total']):>15s}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--finetuned_ckpt", type=str, required=True,
                   help="EditCLIP+E-MMDiT transformer.pt for method A.")
    p.add_argument("--editclip_path", type=str, required=True,
                   help="EditCLIP visual encoder dir for method A.")
    p.add_argument("--task_prompts", type=str, required=True,
                   help="task_prompts.json for method B (text instructions).")
    p.add_argument("--output_json", type=str,
                   default="output/benchmark_compare.json")
    # benchmark knobs
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=5)
    # model (B's Nitro-E base — A uses --finetuned_ckpt)
    p.add_argument("--repo_name", type=str, default="amd/Nitro-E")
    p.add_argument("--ckpt_name", type=str, default="Nitro-E-512px.safetensors")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    # RF-Solver
    p.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    p.add_argument("--delta_t", type=float, default=0.01)
    p.add_argument("--num_inversion_steps", type=int, default=30)
    p.add_argument("--num_denoising_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--llama_max_length", type=int, default=128)
    # PnP
    p.add_argument("--pnp_layer_start", type=int, default=4)
    p.add_argument("--pnp_layer_end", type=int, default=20)
    p.add_argument("--only", type=str, choices=["A", "B", "both"], default="both",
                   help="Run only one method (useful for debugging).")
    args = p.parse_args()

    with open(args.task_prompts) as f:
        raw = json.load(f)
    task_prompts = {k: v for k, v in raw.items() if not k.startswith("_")}

    n_needed = args.warmup + args.measure
    samples = pick_inputs(args.benchmark_root, n_needed)
    print(f"[setup] using {n_needed} (q, ex) pairs from "
          f"{samples[0][2].category}/{samples[0][2].name}")

    result = {"args": vars(args)}
    if args.only in {"A", "both"}:
        result["A"] = profile_editclip(args, samples)
    if args.only in {"B", "both"}:
        result["B"] = profile_llama(args, samples, task_prompts)

    if args.only == "both":
        print_comparison(result["A"], result["B"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".",
                  exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
