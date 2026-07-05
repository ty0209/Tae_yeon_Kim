#!/usr/bin/env bash
# Run both channel-concat methods on the TOP benchmark, then aggregate
# all 4 metrics (LPIPS, SSIM, EC2EC, CLIP) via compare.py.
#
# - EditCLIP + E-MMDiT + concat: papermatch ckpt (Stage 2).
#     Uses EditCLIP paper-style dual CFG: g=7, img_g=1.5.
# - Llama + E-MMDiT + concat: Stage 1 ckpt with task_prompts.json.
#     Single-CFG text guidance: g=4.5.
#
# Fill in the variables below before running.

set -e

# ============ PATHS — EDIT THESE ============
PY=/root/e_mmdit/venv/bin/python
ROOT=/root/e_mmdit/flash-attention/Nitro-E-main
BENCH=$ROOT/benchmark

# Stage 2 (EditCLIP + concat) papermatch checkpoint.
EC_CKPT=$ROOT/output/emmdit_ip2p_stage2_papermatch/2026_05_17-06_56_42_lr2e-05/checkpoint-55000/transformer.pt

# Stage 1 (Llama + concat) checkpoint.
TEXT_CKPT=$ROOT/output/emmdit_ip2p_stage1/2026_05_16-02_49_36_lr5e-05/checkpoint-100000/transformer.pt

# EditCLIP visual encoder (used for cond AND for EC2EC metric).
EDITCLIP=/root/e_mmdit/flash-attention/EditCLIP/clip_ckpt/editclip_vit_l_14

TASK_PROMPTS=$ROOT/topbench_eval/task_prompts.json

OUT=$ROOT/output/topbench_concat
mkdir -p $OUT

# ============ INFERENCE ============

# 1) EditCLIP + E-MMDiT (channel concat) — paper-style dual CFG (g=7, img_g=1.5)
$PY $ROOT/topbench_eval/run_emmdit_concat_editclip.py \
    --benchmark_root $BENCH \
    --finetuned_ckpt $EC_CKPT \
    --editclip_path $EDITCLIP \
    --output_dir $OUT/editclip_concat \
    --resolution 512 \
    --k_exemplars 3 --num_replicates 1 \
    --cfg_mode dual --cfg_text 7.0 --cfg_image 1.5 \
    --num_steps 30 --dtype bf16 \
    --clean_output_dir

# 2) Llama + E-MMDiT (channel concat) — single-CFG text guidance
$PY $ROOT/topbench_eval/run_emmdit_concat_text.py \
    --benchmark_root $BENCH \
    --finetuned_ckpt $TEXT_CKPT \
    --task_prompts $TASK_PROMPTS \
    --output_dir $OUT/text_concat \
    --resolution 512 \
    --num_steps 30 --cfg_scale 4.5 --dtype bf16 \
    --clean_output_dir

# ============ AGGREGATE METRICS ============

# CLIP score paper convention is raw cosine (0-1, EditCLIP table shows ~0.21).
# x100 if you want T2I convention.
$PY $ROOT/topbench_eval/compare.py \
    --run editclip+emmdit+concat=$OUT/editclip_concat \
    --run llama+emmdit+concat=$OUT/text_concat \
    --editclip_path $EDITCLIP \
    --task_prompts $TASK_PROMPTS \
    --output_csv $OUT/compare.csv \
    --output_json $OUT/compare.json \
    --clip_score_scale raw \
    --eval_size 256

echo
echo "[done] Results: $OUT/compare.csv"
echo "       Per-sample: $OUT/compare.json"
