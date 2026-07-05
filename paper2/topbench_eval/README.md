# topbench_eval — EditCLIP-style benchmark comparison

Compare image-editing methods on the EditCLIP
`01/02/03 × task × {train,test}` benchmark. Supported methods:

- **EditCLIP + IP2P** (SD-1.5 + InstructPix2Pix, EditCLIP visual conditioning)
- **EditCLIP + E-MMDiT + PnP** (Nitro-E + RF-Solver + plug-and-play Q/K injection)
- **Llama-3.2 + E-MMDiT + PnP** (text-conditioned variant of the above, same
  backbone, same editing technique, different encoder)

The text method uses per-task prompts from `task_prompts.json`. The fair
comparison: encoder is the only thing that changes between the second and
third methods; CFG / inversion / PnP knobs stay identical.

Metrics:

| metric     | direction | what it measures                                                          |
| ---------- | --------- | ------------------------------------------------------------------------- |
| LPIPS      | ↓ better  | perceptual distance between generated edit and ground-truth target        |
| SSIM       | ↑ better  | structural similarity between generated edit and ground-truth target      |
| EC2EC      | ↑ better  | cosine sim of EditCLIP pair-features: (input, GT-target) vs (input, output) |
| CLIP Score | ↑ better  | cosine sim between target text prompt and model output (×100 by convention) |

EC2EC is EditCLIP's signature edit-fidelity metric — it asks "does the
applied transformation MATCH what the exemplar showed?", not just whether
pixels agree. CLIP Score lets you compare a text-conditioned method against
an exemplar-conditioned method **under a shared text reference** — even
though the exemplar method never sees the text.

## Files

- `dataset.py` — walks `benchmark/<cat>/<task>/{train,test}` and yields
  `EvalSample` rows (test query + K random train exemplars × N replicates).
- `metrics.py` — `lpips_pair`, `ssim_pair`, `ec2ec_pair`, `clip_score`, `compute_all`.
- `task_prompts.json` — per-task source/target/CLIP prompts for the text method
  and for CLIP Score evaluation of the exemplar methods. Edit freely.
- `run_ip2p.py` — runs EditCLIP+IP2P over the benchmark; dumps output PNGs +
  `manifest.json`.
- `run_emmdit_pnp.py` — runs EditCLIP+E-MMDiT+PnP via `inference_nitroE_rfsolver5`.
- `run_text_pnp.py` — runs Llama+E-MMDiT+PnP via `inference_nitroE_rfsolver_text_pnp`.
- `compare.py` — reads one or more manifests and produces a comparison CSV/JSON.

## Pipeline (3 commands)

```bash
EC=/root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14
EC_FT=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_omnistyle_resumed/2026_05_13-04_01_34/checkpoint-100000
EMMDIT_CKPT=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_emmdit_mlp_adapter/2026_05_14-01_10_06_lr5e-05/checkpoint-102000/transformer.pt
BENCH=/root/e_mmdit/flash-attention/Nitro-E-main/benchmark
OUT=/root/e_mmdit/flash-attention/Nitro-E-main/output/topbench

# 1) Run EditCLIP + IP2P
/root/e_mmdit/venv/bin/python topbench_eval/run_ip2p.py \
    --benchmark_root $BENCH \
    --ip2p_ckpt_dir /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_ip2p \
    --editclip_path $EC \
    --output_dir $OUT/ip2p \
    --k_exemplars 3 --num_replicates 1 --resolution 256

# 2) Run EditCLIP + E-MMDiT + PnP
/root/e_mmdit/venv/bin/python topbench_eval/run_emmdit_pnp.py \
    --benchmark_root $BENCH \
    --finetuned_ckpt $EMMDIT_CKPT \
    --editclip_path $EC_FT \
    --output_dir $OUT/emmdit_pnp \
    --k_exemplars 3 --num_replicates 1 \
    --resolution 512 --solver_order 2 \
    --num_inversion_steps 30 --num_denoising_steps 30 \
    --cfg_scale 4.5 \
    --pnp_layer_start 4 --pnp_layer_end 20 \
    --pnp_start_step 0 --pnp_stop_step 30

# 3) (optional) Run Llama + E-MMDiT + PnP — text-conditioned, same backbone
/root/e_mmdit/venv/bin/python topbench_eval/run_text_pnp.py \
    --benchmark_root $BENCH \
    --task_prompts topbench_eval/task_prompts.json \
    --output_dir $OUT/text_pnp \
    --resolution 512 --solver_order 2 \
    --num_inversion_steps 30 --num_denoising_steps 30 \
    --cfg_scale 4.5 \
    --pnp_layer_start 4 --pnp_layer_end 20

# 4) Compute metrics + 3-way comparison CSV
/root/e_mmdit/venv/bin/python topbench_eval/compare.py \
    --run "editclip+ip2p=$OUT/ip2p" \
    --run "editclip+emmdit+pnp=$OUT/emmdit_pnp" \
    --run "llama+emmdit+pnp=$OUT/text_pnp" \
    --editclip_path $EC \
    --task_prompts topbench_eval/task_prompts.json \
    --output_csv $OUT/compare.csv \
    --eval_size 256
```

For a head-to-head **encoder ablation** (same Nitro-E backbone, only encoder
swapped), compare just the second and third runs:

```bash
python topbench_eval/compare.py \
    --run "editclip+emmdit+pnp=$OUT/emmdit_pnp" \
    --run "llama+emmdit+pnp=$OUT/text_pnp" \
    --editclip_path $EC \
    --task_prompts topbench_eval/task_prompts.json \
    --output_csv $OUT/encoder_ablation.csv
```

## Output CSV layout

Rows are `(method, category/task)` with these aggregates at the top of
each method block:

- `ALL/ALL` — overall mean across the whole benchmark
- `01/ALL`, `02/ALL`, `03/ALL` — per-category mean
- `01/boy2girl`, ... — per-task means

Columns: `n, lpips_mean ± std, ssim_mean ± std, ec2ec_mean ± std,
clip_mean ± std, clip_n`. CLIP Score is populated only when `--task_prompts`
is provided (or when the manifest entries embed prompts, as `run_text_pnp.py`
does by default).

### Encoder comparison interpretation

When comparing `editclip+emmdit+pnp` vs `llama+emmdit+pnp`:

- **LPIPS / SSIM** — structure preservation. Both methods share the same
  Nitro-E + RF-Solver + PnP backbone, so deltas here measure how well each
  encoder's conditioning preserves source structure.
- **EC2EC** — does the applied edit match the *exemplar's* edit?
  `editclip` should win here by construction (it sees the exemplar pair;
  `llama` only sees a text approximation).
- **CLIP Score** — does the applied edit match the *target text*?
  Naively `llama` should win (it gets the text directly). If
  `editclip+emmdit+pnp` reaches a comparable CLIP Score *without ever
  receiving the text*, that's a strong claim: exemplar conditioning
  generalises to a text proxy of the same edit.

## Notes / Gotchas

- Both runners save outputs at their **native resolution** (IP2P=256,
  E-MMDiT=512). `compare.py` resizes both to `--eval_size` (default 256) before
  scoring, matching the EditCLIP paper protocol.
- `--k_exemplars 3` averages the EditCLIP visual embedding across 3 random
  train pairs, which is what the EditCLIP paper effectively does. The first
  exemplar's images are also passed to IP2P's `(input_image, edited_image)`
  arguments since the pipeline expects a single concrete pair (the visual
  prompt embedding is what matters for transfer accuracy).
- For the E-MMDiT runner, the source-recording pass shares the same xT
  with the edit pass — RF-Solver's inversion-anchor reuse pattern. The PnP
  controller cache is cleared between samples to free GPU memory.
- If you've trained a Stage-1 adapter-only checkpoint, point `--finetuned_ckpt`
  at the matching `transformer.pt`; the adapter is auto-loaded from the same
  directory's `editclip_adapter.pt` (see `inference_nitroE_rfsolver5.load_models`).
- `--max_tests_per_task N` is useful for a quick smoke run before committing
  to the full benchmark.

## Quick smoke test

```bash
# 1 task, 1 test, 1 exemplar, 1 replicate — should finish in a couple minutes.
/root/e_mmdit/venv/bin/python topbench_eval/run_ip2p.py \
    --benchmark_root $BENCH \
    --ip2p_ckpt_dir /root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_ip2p \
    --editclip_path $EC \
    --output_dir /tmp/smoke_ip2p \
    --categories 01 --tasks boy2girl --max_tests_per_task 1 \
    --k_exemplars 1 --num_replicates 1
```
