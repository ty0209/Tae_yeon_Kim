"""
Run EditCLIP + E-MMDiT + PnP (RF-Solver inference) over the EditCLIP-style
benchmark and dump per-sample output PNGs.

Imports inference logic from `inference_nitroE_rfsolver5.py` so the
generation path is byte-identical to what you would get by invoking that
script directly.

Usage
-----
python topbench_eval/run_emmdit_pnp.py \
    --benchmark_root /root/e_mmdit/flash-attention/Nitro-E-main/benchmark \
    --finetuned_ckpt /path/to/transformer.pt \
    --editclip_path /path/to/editclip_vit_l_14_ft \
    --output_dir /root/e_mmdit/flash-attention/Nitro-E-main/output/topbench/emmdit_pnp \
    --k_exemplars 3 --num_replicates 1 \
    --resolution 512 --solver_order 2 \
    --num_inversion_steps 30 --num_denoising_steps 30 \
    --cfg_scale 4.5 \
    --pnp_layer_start 4 --pnp_layer_end 20 \
    --pnp_start_step 0 --pnp_stop_step 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Make the parent dir importable so we can grab inference_nitroE_rfsolver5
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
from PIL import Image
from tqdm import tqdm

# Re-use the inference primitives from rfsolver5 verbatim.
import inference_nitroE_rfsolver5 as rfs5  # noqa: E402

# Repo-local dataset iterator
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as bench_ds  # noqa: E402


def load_pil(path: str, resolution: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((resolution, resolution),
                                                    Image.LANCZOS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--finetuned_ckpt", type=str, required=True)
    p.add_argument("--editclip_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    # benchmark sampling
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--k_exemplars", type=int, default=3)
    p.add_argument("--num_replicates", type=int, default=1)
    p.add_argument("--max_tests_per_task", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_output_dir", action="store_true",
                   help="Delete and recreate --output_dir before running. "
                        "Prevents stale PNGs from a previous run with "
                        "different settings from coexisting with new ones.")
    # RF-Solver knobs
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    p.add_argument("--delta_t", type=float, default=0.01)
    p.add_argument("--num_inversion_steps", type=int, default=30)
    p.add_argument("--num_denoising_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    # PnP knobs
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
    args = p.parse_args()

    if args.clean_output_dir and os.path.isdir(args.output_dir):
        print(f"[clean] removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # ----- discover tasks -----
    tasks = bench_ds.discover_tasks(args.benchmark_root,
                                      categories=args.categories,
                                      task_filter=args.tasks)
    total = bench_ds.estimate_total_samples(
        tasks, num_replicates=args.num_replicates,
        max_tests_per_task=args.max_tests_per_task,
    )
    print(f"[bench] {len(tasks)} tasks discovered, "
          f"~{total} eval samples (k={args.k_exemplars}, "
          f"replicates={args.num_replicates})")

    # ----- load E-MMDiT + EditCLIP + VAE -----
    (transformer, adapter, vae, editclip_encoder, editclip_processor,
     scheduler) = rfs5.load_models(
        args.finetuned_ckpt, args.editclip_path,
        args.resolution, device, dtype,
    )

    pnp_stop_step = (args.pnp_stop_step if args.pnp_stop_step is not None
                     else args.num_denoising_steps)

    # Install PnP hooks ONCE; flip its mode per sample. Cache is keyed only by
    # block_idx / step_idx, so we must clear the dict between samples.
    pnp_controller = None
    if args.pnp_enable:
        pnp_controller = rfs5.PnPController(
            layer_start=args.pnp_layer_start,
            layer_end=args.pnp_layer_end,
            step_start=args.pnp_start_step,
            step_end=pnp_stop_step,
            inject_q=args.pnp_inject_q,
            inject_k=args.pnp_inject_k,
            verbose=False,
        )
        n_blocks = pnp_controller.install_hooks(transformer)
        print(f"[pnp] hooks installed on {n_blocks} blocks; "
              f"layers [{args.pnp_layer_start},{args.pnp_layer_end}), "
              f"steps [{args.pnp_start_step},{pnp_stop_step})")

    # ----- iterate samples -----
    manifest = []
    pbar = tqdm(bench_ds.iter_samples(
                    tasks,
                    k_exemplars=args.k_exemplars,
                    num_replicates=args.num_replicates,
                    seed=args.seed,
                    max_tests_per_task=args.max_tests_per_task,
                ),
                 total=total, desc="emmdit_pnp")

    for sample in pbar:
        # ---- conditioning: average adapter embeddings over K exemplars ----
        edit_embeds_list = []
        for ex in sample.exemplars:
            ex_in = load_pil(ex.src_path, args.resolution)
            ex_ed = load_pil(ex.ed_path, args.resolution)
            e = rfs5.compute_edit_embeds(
                ex_in, ex_ed,
                editclip_encoder, editclip_processor, adapter,
                device, dtype, adapter_scale=args.adapter_scale,
            )
            edit_embeds_list.append(e)
        edit_embeds = torch.stack(edit_embeds_list, dim=0).mean(dim=0)
        null_embeds = torch.zeros_like(edit_embeds)

        # ---- query -> latent ----
        query_pil = load_pil(sample.query.src_path, args.resolution)
        from torchvision import transforms
        query_tensor = transforms.ToTensor()(query_pil).unsqueeze(0)
        x0 = rfs5.encode_to_latent(query_tensor, vae, device, dtype)

        # ---- inversion (null cond) ----
        xT = rfs5.rfsolver_inversion(
            transformer, scheduler, x0,
            null_embeds=null_embeds, dummy_embeds=null_embeds,
            num_steps=args.num_inversion_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            device=device,
        )

        # ---- source recording pass (null cond) ----
        if pnp_controller is not None:
            pnp_controller.cache.clear()
            pnp_controller.set_mode("record")
            pnp_controller.reset_counters()
        rfs5.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=null_embeds, null_embeds=null_embeds,
            num_steps=args.num_denoising_steps,
            delta_norm=args.delta_t,
            order=args.solver_order,
            cfg_scale=1.0,
            device=device,
            pnp_controller=pnp_controller,
            desc="src-rec",
        )

        # ---- edit pass (edit cond + CFG, PnP inject) ----
        if pnp_controller is not None:
            pnp_controller.set_mode("inject")
        x0_edit = rfs5.rfsolver_denoising(
            transformer, scheduler, xT,
            cond_embeds=edit_embeds, null_embeds=null_embeds,
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

        edit_pixels = rfs5.decode_from_latent(x0_edit, vae).cpu()
        out_pil = rfs5.to_pil(edit_pixels)

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

    if pnp_controller is not None:
        pnp_controller.remove_hooks()

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"method": "editclip+emmdit+pnp",
                    "args": vars(args), "samples": manifest}, f, indent=2)
    print(f"[done] {len(manifest)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
