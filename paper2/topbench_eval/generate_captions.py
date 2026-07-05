"""
Generate per-pair edit-description captions for the EditCLIP TOP-Bench-X
benchmark using LLaVA-1.6 (mistral or vicuna variants).

Each (src, ed) train pair is concatenated side-by-side into a single
canvas. We ask LLaVA to describe how the LEFT image was edited to produce
the RIGHT image. The resulting captions feed run_vlm_text_pnp.py as a fair
text-based counterpart to the EditCLIP visual exemplar pathway:

    EditCLIP path:  (src, ed) -> ViT(6-ch concat) -> 256x2048 visual features
    VLM    path:    (src, ed) -> LLaVA           -> caption     -> Llama-3.2 hidden states

Both pipelines see the SAME train pairs and feed into the SAME Nitro-E +
RF-Solver + PnP backbone. The only thing that changes is the encoder route.

Output
------
A JSON file keyed by "<category>/<task>/<pair_id>" -> caption string:
    {
      "01/boy2girl/243": "A young boy ... is replaced by a young girl ...",
      "01/boy2girl/204": "...",
      ...
    }

Usage
-----
python topbench_eval/generate_captions.py \
    --benchmark_root /root/e_mmdit/flash-attention/Nitro-E-main/benchmark \
    --output_json    topbench_eval/captions_llava16.json \
    --splits train test \
    --vlm_model llava-hf/llava-v1.6-mistral-7b-hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
from PIL import Image
from tqdm import tqdm

import dataset as bench_ds  # noqa: E402


_PROMPT_TEMPLATE = (
    "[INST] <image>\n"
    "The image above is a side-by-side comparison. On the LEFT is the "
    "ORIGINAL image. On the RIGHT is the EDITED version of the same scene. "
    "In one or two sentences, describe the edit that was applied to transform "
    "the left image into the right image. Focus on what changed (subject, "
    "style, color, environment, added objects, removed objects, etc.). "
    "Do not describe what stayed the same. Write the description in present "
    "tense, beginning with 'The edit'."
    " [/INST]"
)


def side_by_side(src: Image.Image, ed: Image.Image, h: int = 384) -> Image.Image:
    """Concatenate two images side-by-side at a common height `h`.
    Keeps aspect ratios. LLaVA-1.6 downsamples internally to ~672 long edge.
    """
    src = src.convert("RGB")
    ed = ed.convert("RGB")
    ws = max(1, int(src.width * (h / src.height)))
    we = max(1, int(ed.width * (h / ed.height)))
    src_r = src.resize((ws, h), Image.LANCZOS)
    ed_r = ed.resize((we, h), Image.LANCZOS)
    out = Image.new("RGB", (ws + we, h), color=(255, 255, 255))
    out.paste(src_r, (0, 0))
    out.paste(ed_r, (ws, 0))
    return out


def load_llava(model_name: str, device: torch.device, dtype: torch.dtype):
    """Returns (processor, model). Supports `llava-hf/llava-v1.6-*` HF ids."""
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    print(f"[load] {model_name}")
    proc = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    return proc, model


@torch.no_grad()
def caption_pair(model, proc, src: Image.Image, ed: Image.Image,
                  device: torch.device, max_new_tokens: int,
                  prompt_template: str) -> str:
    canvas = side_by_side(src, ed)
    inputs = proc(images=canvas, text=prompt_template, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                          do_sample=False, num_beams=1)
    text = proc.decode(out[0], skip_special_tokens=True)
    # Strip the prompt portion ([INST] ... [/INST]) — keep only the answer.
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[1]
    return text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_root", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="+", default=["train"],
                   choices=["train", "test"],
                   help="Which split(s) to caption. 'train' is the only one "
                        "needed for the standard exemplar protocol; add "
                        "'test' if you want to caption test pairs too.")
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=None)
    p.add_argument("--vlm_model", type=str,
                   default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--resume", action="store_true",
                   help="If output_json exists, load it and SKIP pairs that "
                        "already have a caption. Useful for incremental runs.")
    p.add_argument("--prompt_template", type=str, default=_PROMPT_TEMPLATE,
                   help="Override the [INST]...[/INST] prompt template.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[args.dtype]

    # ----- discover pairs -----
    tasks = bench_ds.discover_tasks(args.benchmark_root,
                                      categories=args.categories,
                                      task_filter=args.tasks)
    work_items: List[Tuple[str, bench_ds.Pair]] = []
    for task in tasks:
        if "train" in args.splits:
            for pair in task.train_pairs:
                key = f"{task.category}/{task.name}/{pair.pair_id}"
                work_items.append((key, pair))
        if "test" in args.splits:
            for pair in task.test_pairs:
                key = f"{task.category}/{task.name}/{pair.pair_id}"
                work_items.append((key, pair))
    print(f"[bench] {len(tasks)} tasks, {len(work_items)} pairs to caption")

    # ----- resume -----
    existing: dict = {}
    if args.resume and os.path.exists(args.output_json):
        with open(args.output_json) as f:
            existing = json.load(f)
        work_items = [(k, p) for k, p in work_items if k not in existing]
        print(f"[resume] {len(existing)} captions already present; "
              f"{len(work_items)} remaining")

    if not work_items:
        print("[done] nothing to do.")
        return

    # ----- load VLM -----
    proc, model = load_llava(args.vlm_model, device, dtype)

    # ----- caption -----
    out_dict = dict(existing)
    try:
        for key, pair in tqdm(work_items, desc="caption"):
            src = Image.open(pair.src_path)
            ed = Image.open(pair.ed_path)
            try:
                caption = caption_pair(model, proc, src, ed, device,
                                          args.max_new_tokens,
                                          args.prompt_template)
            except Exception as e:
                print(f"[err] {key}: {e}")
                continue
            out_dict[key] = caption
            # Periodic flush so a crash doesn't lose everything.
            if len(out_dict) % 25 == 0:
                _save(args.output_json, out_dict)
    finally:
        _save(args.output_json, out_dict)
    print(f"[done] wrote {len(out_dict)} captions -> {args.output_json}")


def _save(path: str, d: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
