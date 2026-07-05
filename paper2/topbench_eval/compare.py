"""
Compute LPIPS / SSIM / EC2EC / CLIP Score from one or more run output dirs
and emit a comparison CSV + JSON summary across methods.

Each run dir must contain a `manifest.json` produced by run_ip2p.py,
run_emmdit_pnp.py, or run_text_pnp.py. Each entry in the manifest tells us
where the model output lives and which test ground-truth to compare against.

CLIP Score is computed per sample against the task's `clip_prompt` (falls
back to `target_prompt`). Prompts come from EITHER the manifest entry (if
the runner stored them, as run_text_pnp.py does) OR a separate
`--task_prompts` JSON keyed by "<category>/<task>" — so the EditCLIP /
IP2P methods (which never saw a prompt) still get CLIP-scored against the
same prompt as the Llama method.

Usage
-----
python topbench_eval/compare.py \
    --run editclip+ip2p=/path/to/output/topbench/ip2p \
    --run editclip+emmdit+pnp=/path/to/output/topbench/emmdit_pnp \
    --run llama+emmdit+pnp=/path/to/output/topbench/text_pnp \
    --editclip_path /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14 \
    --task_prompts topbench_eval/task_prompts.json \
    --output_csv /path/to/output/topbench/compare.csv \
    --eval_size 256

`--run NAME=PATH` may be passed multiple times.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import metrics as M  # noqa: E402


def parse_run_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"--run expects NAME=PATH, got {s!r}")
    name, path = s.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not os.path.isdir(path):
        raise FileNotFoundError(f"run dir not found: {path}")
    if not os.path.exists(os.path.join(path, "manifest.json")):
        raise FileNotFoundError(f"manifest.json missing in {path}")
    return name, path


def load_manifest(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "manifest.json")) as f:
        m = json.load(f)
    return m.get("samples", [])


def resolve_clip_prompt(entry: dict, task_prompts: dict) -> str:
    """Pull clip_prompt for this entry. Priority: manifest entry's
    explicit clip_prompt -> manifest's target_prompt -> task_prompts.json
    lookup by category/task -> None.
    """
    if "clip_prompt" in entry and entry["clip_prompt"]:
        return entry["clip_prompt"]
    if "target_prompt" in entry and entry["target_prompt"]:
        return entry["target_prompt"]
    key = f"{entry['category']}/{entry['task']}"
    info = task_prompts.get(key)
    if info is None:
        return ""
    return info.get("clip_prompt") or info.get("target_prompt") or ""


def score_one(entry: dict, editclip_path: str, eval_size: int,
                clip_prompt: str = "",
                clip_model_name: str = "openai/clip-vit-large-patch14",
                clip_score_scale: str = "x100") -> dict:
    in_pil = Image.open(entry["query_src"]).convert("RGB")
    gt_pil = Image.open(entry["query_gt_edit"]).convert("RGB")
    out_pil = Image.open(entry["output"]).convert("RGB")

    # EditCLIP-paper EC2EC needs an exemplar pair. Methods that store
    # `exemplars` in the manifest (run_emmdit_pnp / run_ip2p / run_vlm_text_pnp)
    # populate this; text_pnp's manifest has no exemplars so the metric is
    # skipped for that method (only the biased ec2ec_to_gt is reported).
    exemplar_src_pil = None
    exemplar_ed_pil = None
    exemplars = entry.get("exemplars") or []
    if exemplars:
        ex = exemplars[0]   # use the first exemplar as the reference pair
        try:
            exemplar_src_pil = Image.open(ex["src"]).convert("RGB")
            exemplar_ed_pil  = Image.open(ex["ed"]).convert("RGB")
        except Exception:
            exemplar_src_pil = exemplar_ed_pil = None

    out = M.compute_all(in_pil, gt_pil, out_pil,
                          editclip_path=editclip_path, size=eval_size,
                          clip_prompt=clip_prompt or None,
                          clip_model_name=clip_model_name,
                          exemplar_src=exemplar_src_pil,
                          exemplar_ed=exemplar_ed_pil)
    # compute_all() defaults to x100. Down-scale if user wants raw cosine.
    if "clip_score" in out and clip_score_scale == "raw":
        out["clip_score"] = out["clip_score"] / 100.0
    return out


def aggregate(per_sample: List[dict]) -> Dict[str, dict]:
    """Bucket per-sample scores by (method, category, task) -> stats.

    Returns nested dict: { method -> { (category, task) -> dict } }
    with an extra (category="ALL", task="ALL") aggregate.
    """
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in per_sample:
        grouped[(row["method"], row["category"], row["task"])].append(row)

    out: Dict[str, dict] = {}
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for (method, cat, task), rows in grouped.items():
        by_method[method].extend(rows)
        out.setdefault(method, {})[f"{cat}/{task}"] = _stats(rows)

    for method, rows in by_method.items():
        out[method]["ALL/ALL"] = _stats(rows)
        # also per-category aggregates
        by_cat: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        for cat, rs in by_cat.items():
            out[method][f"{cat}/ALL"] = _stats(rs)
    return out


def _stats(rows: List[dict]) -> dict:
    lpips = np.array([r["lpips"] for r in rows], dtype=np.float64)
    ssim = np.array([r["ssim"] for r in rows], dtype=np.float64)
    # Legacy biased-high metric. May be stored under either name depending
    # on the version of metrics.py that produced the rows.
    ec2ec_gt_vals = [r.get("ec2ec_to_gt", r.get("ec2ec")) for r in rows
                      if ("ec2ec_to_gt" in r) or ("ec2ec" in r)]
    ec2ec_gt = (np.array(ec2ec_gt_vals, dtype=np.float64)
                 if ec2ec_gt_vals else np.array([]))
    # EditCLIP paper's edit-direction metric. Only present when exemplar
    # info was available in the manifest entry.
    ec2ec_ed_vals = [r["ec2ec_edit_dir"] for r in rows
                      if "ec2ec_edit_dir" in r]
    ec2ec_ed = (np.array(ec2ec_ed_vals, dtype=np.float64)
                 if ec2ec_ed_vals else np.array([]))
    clip_vals = [r["clip_score"] for r in rows if "clip_score" in r]
    clip = np.array(clip_vals, dtype=np.float64) if clip_vals else np.array([])

    def _ms(arr):
        if not len(arr):
            return float("nan"), float("nan"), 0
        return float(arr.mean()), float(arr.std()), int(len(arr))

    ec_gt_m, ec_gt_s, ec_gt_n = _ms(ec2ec_gt)
    ec_ed_m, ec_ed_s, ec_ed_n = _ms(ec2ec_ed)
    clip_m, clip_s, clip_n = _ms(clip)

    return {
        "n": int(len(rows)),
        "lpips_mean": float(lpips.mean()) if len(lpips) else float("nan"),
        "lpips_std":  float(lpips.std())  if len(lpips) else float("nan"),
        "ssim_mean":  float(ssim.mean())  if len(ssim)  else float("nan"),
        "ssim_std":   float(ssim.std())   if len(ssim)  else float("nan"),
        # EditCLIP paper's metric (sources differ, ~0.3-0.6 range).
        "ec2ec_edit_dir_mean": ec_ed_m,
        "ec2ec_edit_dir_std":  ec_ed_s,
        "ec2ec_edit_dir_n":    ec_ed_n,
        # Legacy biased-high metric (shared source -> ~0.85+).
        "ec2ec_to_gt_mean": ec_gt_m,
        "ec2ec_to_gt_std":  ec_gt_s,
        "ec2ec_to_gt_n":    ec_gt_n,
        "clip_mean": clip_m,
        "clip_std":  clip_s,
        "clip_n":    clip_n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="Repeatable. Format: NAME=PATH_TO_RUN_DIR.")
    ap.add_argument("--editclip_path", type=str, required=True,
                    help="EditCLIP HF dir for the EC2EC metric.")
    ap.add_argument("--task_prompts", type=str, default=None,
                    help="Optional. Path to task_prompts.json. When provided "
                         "(or when manifest entries embed prompts), CLIP "
                         "Score is computed against the per-task prompt.")
    ap.add_argument("--clip_model", type=str,
                    default="openai/clip-vit-large-patch14",
                    help="CLIP model id for CLIP Score (text-image cosine). "
                         "Distinct from --editclip_path which is for EC2EC.")
    ap.add_argument("--clip_score_scale", type=str, default="x100",
                    choices=["raw", "x100"],
                    help="Reporting convention for CLIP Score. 'raw' = "
                         "cosine similarity in [0, 1] (matches the EditCLIP "
                         "paper, e.g. 0.216). 'x100' = cosine x 100 in "
                         "[0, 100] (some T2I papers). Default x100.")
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--output_json", type=str, default=None,
                    help="Optional. Defaults to alongside --output_csv.")
    ap.add_argument("--eval_size", type=int, default=256)
    ap.add_argument("--lpips_net", type=str, default="alex",
                    choices=["alex", "vgg", "squeeze"])
    args = ap.parse_args()

    runs = [parse_run_arg(s) for s in args.run]

    # Optional task prompt map (so methods that don't store prompts in their
    # manifest can still get a CLIP Score against the same per-task text).
    task_prompts: dict = {}
    if args.task_prompts:
        with open(args.task_prompts) as f:
            raw = json.load(f)
        task_prompts = {k: v for k, v in raw.items() if not k.startswith("_")}
        print(f"[prompts] loaded {len(task_prompts)} per-task prompts")

    # ---- score every (method, sample) ----
    per_sample: List[dict] = []
    for method_name, run_dir in runs:
        samples = load_manifest(run_dir)
        print(f"[score] {method_name}: {len(samples)} samples from {run_dir}")
        for entry in tqdm(samples, desc=method_name):
            clip_prompt = resolve_clip_prompt(entry, task_prompts)
            try:
                m = score_one(entry, editclip_path=args.editclip_path,
                                eval_size=args.eval_size,
                                clip_prompt=clip_prompt,
                                clip_model_name=args.clip_model,
                                clip_score_scale=args.clip_score_scale)
            except FileNotFoundError as e:
                # Skip rows whose output PNG was not produced (e.g. crashed run).
                print(f"[skip] {entry.get('output')}: {e}")
                continue
            per_sample.append({
                "method":   method_name,
                "category": entry["category"],
                "task":     entry["task"],
                "query_id": entry["query_id"],
                "replicate": entry.get("replicate", 0),
                "output":   entry["output"],
                "clip_prompt": clip_prompt,
                **m,
            })

    # ---- aggregate ----
    agg = aggregate(per_sample)

    # ---- write CSV ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        clip_label = ("clip_mean(raw)" if args.clip_score_scale == "raw"
                       else "clip_mean(x100)")
        clip_std_label = ("clip_std(raw)" if args.clip_score_scale == "raw"
                           else "clip_std(x100)")
        # EditCLIP paper terminology: SSIM is reported as S_visual.
        w.writerow(["method", "category/task", "n",
                     "lpips_mean", "lpips_std",
                     "s_visual_mean", "s_visual_std",
                     # Paper's EC2EC: edit-direction (sources differ).
                     "ec2ec_edit_dir_mean", "ec2ec_edit_dir_std",
                     "ec2ec_edit_dir_n",
                     # Biased-high diagnostic: cosine with shared source.
                     "ec2ec_to_gt_mean", "ec2ec_to_gt_std",
                     clip_label, clip_std_label, "clip_n"])
        for method, buckets in agg.items():
            for key, st in sorted(buckets.items(), key=_bucket_sort_key):
                w.writerow([method, key, st["n"],
                             f"{st['lpips_mean']:.4f}", f"{st['lpips_std']:.4f}",
                             f"{st['ssim_mean']:.4f}",  f"{st['ssim_std']:.4f}",
                             f"{st['ec2ec_edit_dir_mean']:.4f}",
                             f"{st['ec2ec_edit_dir_std']:.4f}",
                             st["ec2ec_edit_dir_n"],
                             f"{st['ec2ec_to_gt_mean']:.4f}",
                             f"{st['ec2ec_to_gt_std']:.4f}",
                             f"{st['clip_mean']:.4f}",  f"{st['clip_std']:.4f}",
                             st["clip_n"]])
    print(f"[done] CSV -> {args.output_csv}")

    # ---- write JSON ----
    out_json = args.output_json or (os.path.splitext(args.output_csv)[0] + ".json")
    with open(out_json, "w") as f:
        json.dump({"per_sample": per_sample, "aggregated": agg},
                   f, indent=2)
    print(f"[done] JSON -> {out_json}")

    # ---- pretty print method overall lines (EditCLIP paper labels) ----
    print("\n=== summary (ALL/ALL) ===")
    print(f"{'method':<28} {'n':>5} {'LPIPS':>8} {'S_visual':>9} "
           f"{'EC2EC':>9} {'EC2EC_gt':>10} {'CLIP':>9}")
    for method, buckets in agg.items():
        st = buckets.get("ALL/ALL")
        if st is None:
            continue
        print(f"{method:<28} {st['n']:>5} "
               f"{st['lpips_mean']:>8.4f} {st['ssim_mean']:>9.4f} "
               f"{st['ec2ec_edit_dir_mean']:>9.4f} "
               f"{st['ec2ec_to_gt_mean']:>10.4f} "
               f"{st['clip_mean']:>9.4f}")
    print("  S_visual  = SSIM (EditCLIP paper terminology)")
    print("  EC2EC     = EditCLIP paper convention (exemplar pair vs output pair)")
    print("  EC2EC_gt  = diagnostic only — test_input shared both sides (biased high)")


def _bucket_sort_key(item):
    """Sort so ALL/ALL is first, then xx/ALL, then individual tasks alphabetic."""
    key, _ = item
    cat, task = key.split("/", 1)
    return (0 if cat == "ALL" else 1,
             0 if task == "ALL" else 1,
             cat, task)


if __name__ == "__main__":
    main()
