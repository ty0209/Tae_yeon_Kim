"""
Run Llama-3.2 + Nitro-E + RF-Solver + PnP over the EditCLIP-style benchmark.

Text instructions come from task_prompts.json (per-task source/target prompts).
This is the text-conditioned counterpart of run_emmdit_pnp.py — same model
backbone (Nitro-E + RF-Solver + PnP), different conditioning source.

NOTE: this method does NOT use any exemplars. The task name selects the prompt;
the test query image is what gets edited.

Usage
-----
python topbench_eval/run_text_pnp.py \
    --benchmark_root .../benchmark \
    --task_prompts topbench_eval/task_prompts.json \
    --output_dir .../output/topbench/text_pnp \
    --resolution 512 --solver_order 2 \
    --num_inversion_steps 30 --num_denoising_steps 30 \
    --cfg_scale 4.5 \
    --pnp_layer_start 4 --pnp_layer_end 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
from PIL import Image
from tqdm import tqdm

# Reuse the text+PnP inference helpers verbatim.
import inference_nitroE_rfsolver_text_pnp as rfst_pnp  # noqa: E402
from inference_nitroE_rfsolver_text import (  # noqa: E402
    load_models, encode_prompt, load_image,
    encode_to_latent, decode_from_latent, to_pil,
)
from inference_nitroE_rfsolver5 import PnPController  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as bench_ds  # noqa: E402


def load_task_prompts(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--task_prompts", type=str, required=True,
                   help="Path to task_prompts.json.")
    p.add_argument("--output_dir", type=str, required=True)
    # benchmark sampling
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--num_replicates", type=int, default=1,
                   help="Multiple seeds per test for averaging. The text "
                        "method has no exemplars to vary, so replicates "
                        "just vary noise/seed.")
    p.add_argument("--max_tests_per_task", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_output_dir", action="store_true",
                   help="Delete and recreate --output_dir before running.")
    # model + RF-Solver
    p.add_argument("--repo_name", type=str, default="amd/Nitro-E")
    p.add_argument("--ckpt_name", type=str, default="Nitro-E-512px.safetensors")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    p.add_argument("--delta_t", type=float, default=0.01)
    p.add_argument("--num_inversion_steps", type=int, default=30)
    p.add_argument("--num_denoising_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--cfg_scale_invert", type=float, default=1.0)
    p.add_argument("--llama_max_length", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    # PnP
    p.add_argument("--pnp_enable", action="store_true", default=True)
    p.add_argument("--no_pnp", dest="pnp_enable", action="store_false")
    p.add_argument("--pnp_layer_start", type=int, default=4)
    p.add_argument("--pnp_layer_end", type=int, default=20)
    p.add_argument("--pnp_start_step", type=int, default=0)
    p.add_argument("--pnp_stop_step", type=int, default=None)
    p.add_argument("--pnp_inject_q", action="store_true", default=True)
    p.add_argument("--no_pnp_q", dest="pnp_inject_q", action="store_false")
    p.add_argument("--pnp_inject_k", action="store_true", default=True)
    p.add_argument("--no_pnp_k", dest="pnp_inject_k", action="store_false")
    p.add_argument("--source_record_with_source_prompt", action="store_true",
                   default=False)
    args = p.parse_args()

    if args.clean_output_dir and os.path.isdir(args.output_dir):
        print(f"[clean] removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # ----- prompts -----
    prompts = load_task_prompts(args.task_prompts)
    print(f"[prompts] loaded {len(prompts)} task prompts from {args.task_prompts}")

    # ----- tasks -----
    tasks = bench_ds.discover_tasks(args.benchmark_root,
                                      categories=args.categories,
                                      task_filter=args.tasks)
    # Drop any task missing a prompt.
    tasks = [t for t in tasks if f"{t.category}/{t.name}" in prompts]
    total = bench_ds.estimate_total_samples(
        tasks, num_replicates=args.num_replicates,
        max_tests_per_task=args.max_tests_per_task,
    )
    print(f"[bench] {len(tasks)} tasks with prompts; ~{total} eval samples "
          f"(replicates={args.num_replicates})")

    # ----- load Nitro-E + Llama + VAE -----
    tokenizer, text_encoder, vae, transformer, scheduler = load_models(
        args.repo_name, args.ckpt_name, args.resolution, device, dtype,
    )

    # ----- shared null embedding -----
    nul_emb, nul_mask = encode_prompt("", tokenizer, text_encoder,
                                       device, dtype,
                                       max_length=args.llama_max_length)

    # ----- PnP controller -----
    pnp_stop_step = (args.pnp_stop_step if args.pnp_stop_step is not None
                     else args.num_denoising_steps)
    pnp_controller = None
    if args.pnp_enable:
        pnp_controller = PnPController(
            layer_start=args.pnp_layer_start,
            layer_end=args.pnp_layer_end,
            step_start=args.pnp_start_step,
            step_end=pnp_stop_step,
            inject_q=args.pnp_inject_q,
            inject_k=args.pnp_inject_k,
            verbose=False,
        )
        n_blocks = pnp_controller.install_hooks(transformer)
        print(f"[pnp] hooks on {n_blocks} blocks; layers "
              f"[{args.pnp_layer_start},{args.pnp_layer_end}); "
              f"steps [{args.pnp_start_step},{pnp_stop_step})")

    # ----- iterate -----
    manifest = []
    pbar = tqdm(bench_ds.iter_samples(
                    tasks,
                    k_exemplars=1,   # unused for text method
                    num_replicates=args.num_replicates,
                    seed=args.seed,
                    max_tests_per_task=args.max_tests_per_task,
                ),
                 total=total, desc="text_pnp")

    # cache per-task encoded prompts so we don't re-run Llama 100x.
    prompt_cache = {}

    for sample in pbar:
        task_id = f"{sample.task.category}/{sample.task.name}"
        entry = prompts[task_id]
        src_text = entry.get("source_prompt", "")
        tgt_text = entry["target_prompt"]
        clip_text = entry.get("clip_prompt", tgt_text)

        # ---- encode (cached per task) ----
        if task_id not in prompt_cache:
            src_emb, src_mask = encode_prompt(src_text, tokenizer, text_encoder,
                                                device, dtype,
                                                max_length=args.llama_max_length)
            tgt_emb, tgt_mask = encode_prompt(tgt_text, tokenizer, text_encoder,
                                                device, dtype,
                                                max_length=args.llama_max_length)
            prompt_cache[task_id] = (src_emb, src_mask, tgt_emb, tgt_mask)
        src_emb, src_mask, tgt_emb, tgt_mask = prompt_cache[task_id]

        # ---- query -> latent ----
        _query_pil, query_tensor = load_image(sample.query.src_path,
                                                args.resolution)
        x0 = encode_to_latent(query_tensor, vae, device, dtype)

        # ---- inversion ----
        invert_cond, invert_mask = ((src_emb, src_mask) if src_text
                                      else (nul_emb, nul_mask))
        xT = rfst_pnp.rfsolver_inversion(
            transformer, scheduler, x0,
            cond_embeds=invert_cond, cond_mask=invert_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_inversion_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=args.cfg_scale_invert,
            device=device,
            desc="inv",
        )

        # ---- source recording pass ----
        if pnp_controller is not None:
            pnp_controller.cache.clear()
            pnp_controller.set_mode("record")
            pnp_controller.reset_counters()
        rec_cond, rec_mask = ((src_emb, src_mask)
                               if args.source_record_with_source_prompt
                               else (nul_emb, nul_mask))
        rfst_pnp.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=rec_cond, cond_mask=rec_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=1.0,
            device=device,
            pnp_controller=pnp_controller,
            desc="src-rec",
        )

        # ---- edit pass ----
        if pnp_controller is not None:
            pnp_controller.set_mode("inject")
        x0_edit = rfst_pnp.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=tgt_emb, cond_mask=tgt_mask,
            null_embeds=nul_emb, null_mask=nul_mask,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=args.cfg_scale,
            device=device,
            pnp_controller=pnp_controller,
            desc="edit",
        )
        if pnp_controller is not None:
            pnp_controller.set_mode("off")

        out_pil = to_pil(decode_from_latent(x0_edit, vae).cpu())

        # ---- save ----
        task_dir = os.path.join(args.output_dir, sample.task.category,
                                  sample.task.name)
        os.makedirs(task_dir, exist_ok=True)
        out_name = f"{sample.query.pair_id}__rep{sample.sample_idx}.png"
        out_path = os.path.join(task_dir, out_name)
        out_pil.save(out_path)

        manifest.append({
            "category": sample.task.category,
            "task": sample.task.name,
            "query_id": sample.query.pair_id,
            "query_src": sample.query.src_path,
            "query_gt_edit": sample.query.ed_path,
            "source_prompt": src_text,
            "target_prompt": tgt_text,
            "clip_prompt": clip_text,
            "replicate": sample.sample_idx,
            "output": out_path,
        })

    if pnp_controller is not None:
        pnp_controller.remove_hooks()

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"method": "llama+emmdit+pnp",
                    "args": vars(args), "samples": manifest}, f, indent=2)
    print(f"[done] {len(manifest)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
