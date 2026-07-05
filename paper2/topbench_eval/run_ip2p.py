"""
Run EditCLIP + InstructPix2Pix (EditCLIP transfer pipeline) over the
EditCLIP-style benchmark and dump per-sample output PNGs.

This script does ONLY generation. Metrics are computed by compare.py from
the produced output directory.

Usage
-----
python topbench_eval/run_ip2p.py \
    --benchmark_root /root/e_mmdit/flash-attention/Nitro-E-main/benchmark \
    --ip2p_ckpt_dir /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_ip2p \
    --editclip_path /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14 \
    --output_dir /root/e_mmdit/flash-attention/Nitro-E-main/output/topbench/ip2p \
    --k_exemplars 3 --num_replicates 1 --resolution 256
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure EditCLIP's package is importable BEFORE diffusers init.
_EDITCLIP_SRC = "/root/e_mmdit/flash-attention/EditCLIP/src/sd_ip2p_train"
if _EDITCLIP_SRC not in sys.path:
    sys.path.insert(0, _EDITCLIP_SRC)

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from transformers import CLIPModel, AutoProcessor
from diffusers import UNet2DConditionModel

# Locally-importable EditCLIP IP2P pipeline + projection model.
from pipeline_ip2p_transfer import StableDiffusionInstructPix2PixTransferPipeline
from projection_model import ImageProjModel

# Repo-local dataset iterator
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as bench_ds  # noqa: E402


def pil_to_tensor(pil_image, resize):
    img = pil_image.convert("RGB").resize(resize)
    return transforms.ToTensor()(img).unsqueeze(0)


def encode_exemplar(image_encoder, image_processor,
                     in_pil, ed_pil, device, dtype,
                     use_projection: bool, image_proj_model,
                     resolution: int):
    """6-channel concat -> EditCLIP -> (optional projection) -> embeds."""
    in_t = pil_to_tensor(in_pil, (resolution, resolution))
    ed_t = pil_to_tensor(ed_pil, (resolution, resolution))
    in_p = image_processor(images=in_t, return_tensors="pt", do_rescale=False)
    ed_p = image_processor(images=ed_t, return_tensors="pt", do_rescale=False)
    concat = torch.cat([in_p.pixel_values, ed_p.pixel_values], dim=1)
    concat = concat.to(device, dtype=dtype)
    feats = image_encoder(concat)[0]
    if use_projection:
        feats = image_proj_model(feats).to(dtype=dtype)
    uncond = torch.zeros_like(feats)
    return feats, uncond


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--ip2p_ckpt_dir", type=str, required=True,
                   help="Path to editclip_ip2p (contains unet/, image_proj_model.pt).")
    p.add_argument("--editclip_path", type=str, required=True,
                   help="EditCLIP HF visual encoder dir.")
    p.add_argument("--pretrained_sd_path", type=str,
                   default="timbrooks/instruct-pix2pix")
    p.add_argument("--output_dir", type=str, required=True)
    # benchmark sampling
    p.add_argument("--categories", type=str, nargs="+", default=None,
                   help="Restrict categories (e.g. 01 02). Default: all.")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Restrict task names. Default: all.")
    p.add_argument("--k_exemplars", type=int, default=3,
                   help="Train pairs sampled per query (EditCLIP averages over K).")
    p.add_argument("--num_replicates", type=int, default=1,
                   help="How many independent exemplar draws per query.")
    p.add_argument("--max_tests_per_task", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_output_dir", action="store_true",
                   help="Delete and recreate --output_dir before running.")
    # inference
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--use_projection", action="store_true", default=True,
                   help="Apply ImageProjModel (matches editclip_ip2p training).")
    p.add_argument("--no_projection", dest="use_projection",
                   action="store_false")
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    if args.clean_output_dir and os.path.isdir(args.output_dir):
        print(f"[clean] removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

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

    # ----- load models -----
    print(f"[load] UNet from {args.ip2p_ckpt_dir}")
    unet = UNet2DConditionModel.from_pretrained(args.ip2p_ckpt_dir,
                                                  subfolder="unet")
    unet = unet.to(device=device, dtype=dtype)

    print(f"[load] EditCLIP vision_model from {args.editclip_path}")
    image_encoder = CLIPModel.from_pretrained(args.editclip_path).vision_model
    image_encoder = image_encoder.to(device=device, dtype=dtype)
    image_processor = AutoProcessor.from_pretrained(args.editclip_path)

    image_proj_model = None
    if args.use_projection:
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.hidden_size,
            clip_extra_context_tokens=1,
        )
        proj_path = os.path.join(args.ip2p_ckpt_dir, "image_proj_model.pt")
        image_proj_model.load_state_dict(torch.load(proj_path,
                                                     map_location="cpu"))
        image_proj_model = image_proj_model.to(device=device, dtype=dtype)

    pipeline = StableDiffusionInstructPix2PixTransferPipeline.from_pretrained(
        args.pretrained_sd_path, unet=unet, torch_dtype=dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ----- iterate samples -----
    manifest = []
    pbar = tqdm(bench_ds.iter_samples(
                    tasks,
                    k_exemplars=args.k_exemplars,
                    num_replicates=args.num_replicates,
                    seed=args.seed,
                    max_tests_per_task=args.max_tests_per_task,
                ),
                 total=total, desc="ip2p")

    # The IP2P pipeline takes ONE (input_image, edited_image) exemplar.
    # For K>1 exemplars we average the resulting embedding before sending
    # to the pipeline — closer to EditCLIP's paper protocol than running
    # K full pipelines, and far cheaper.
    for sample in pbar:
        query_pil = Image.open(sample.query.src_path).convert("RGB")
        query_pil_eval = query_pil.resize((args.resolution, args.resolution),
                                            Image.LANCZOS)

        embeds = []
        uncs = []
        for ex in sample.exemplars:
            ex_in = Image.open(ex.src_path).convert("RGB")
            ex_ed = Image.open(ex.ed_path).convert("RGB")
            e, u = encode_exemplar(
                image_encoder, image_processor, ex_in, ex_ed,
                device, dtype, args.use_projection, image_proj_model,
                resolution=args.resolution,
            )
            embeds.append(e)
            uncs.append(u)
        prompt_embeds = torch.stack(embeds, dim=0).mean(dim=0)
        uncond = uncs[0]   # zeros — averaging is a no-op

        with torch.no_grad():
            out = pipeline(
                image=query_pil_eval,
                input_image=Image.open(sample.exemplars[0].src_path).convert("RGB"),
                edited_image=Image.open(sample.exemplars[0].ed_path).convert("RGB"),
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=uncond,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

        # ---- save ----
        task_dir = os.path.join(args.output_dir, sample.task.category,
                                  sample.task.name)
        os.makedirs(task_dir, exist_ok=True)
        out_name = (f"{sample.query.pair_id}__rep{sample.sample_idx}"
                    f"__ex-{'-'.join(p.pair_id for p in sample.exemplars)}.png")
        out_path = os.path.join(task_dir, out_name)
        out.save(out_path)

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

    # ----- manifest -----
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"method": "editclip+ip2p", "args": vars(args),
                    "samples": manifest}, f, indent=2)
    print(f"[done] {len(manifest)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
