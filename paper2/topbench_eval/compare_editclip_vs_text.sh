#!/usr/bin/env bash
# Head-to-head comparison on TOP-Bench-X following EditCLIP's eval protocol:
#
#   Method A : EditCLIP + E-MMDiT + PnP   (visual exemplar conditioning,
#               using your fine-tuned Nitro-E + EditCLIP adapter)
#   Method B : Llama + E-MMDiT + PnP      (text instruction conditioning,
#               base Nitro-E + Llama-3.2 text encoder)
#
# Both methods share the same RF-Solver order/steps and PnP window, so
# inference-side hyperparameters are identical. The encoder route is the
# only thing that differs.
#
# Usage:
#   bash topbench_eval/compare_editclip_vs_text.sh
#
# Override defaults via env vars:
#   FINETUNED_CKPT=... EDITCLIP_PATH=... CFG=2.5 ADAPTER_SCALE=1.0 \
#     bash topbench_eval/compare_editclip_vs_text.sh

set -euo pipefail

# ---------- defaults ----------
: "${PY:=/root/e_mmdit/venv/bin/python}"
: "${BENCH:=/root/e_mmdit/flash-attention/Nitro-E-main/benchmark}"
: "${OUT:=/root/e_mmdit/flash-attention/Nitro-E-main/output/topbench}"
: "${FINETUNED_CKPT:=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_emmdit_mlp_adapter/2026_05_14-01_10_06_lr5e-05/checkpoint-102000/transformer.pt}"
: "${EDITCLIP_PATH:=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_omnistyle_resumed/2026_05_13-04_01_34/checkpoint-100000}"
: "${TASK_PROMPTS:=topbench_eval/task_prompts.json}"

# Inference knobs (shared between methods so the encoder route is the ablation).
: "${K_EXEMPLARS:=1}"
: "${CFG:=0.5}"
: "${ADAPTER_SCALE:=0.5}"   # EditCLIP-only knob; text method ignores this.
: "${PNP_LAYER_START:=4}"
: "${PNP_LAYER_END:=20}"
: "${PNP_START_STEP:=0}"
: "${PNP_STOP_STEP:=30}"
: "${NUM_INVERSION_STEPS:=30}"
: "${NUM_DENOISING_STEPS:=30}"
: "${SOLVER_ORDER:=2}"
: "${RESOLUTION:=512}"
: "${DTYPE:=bf16}"

# Optional benchmark filter — leave empty for full benchmark.
: "${CATEGORIES:=}"        # e.g. "01 02"
: "${TASKS:=}"             # e.g. "boy2girl charcoal"
: "${MAX_TESTS_PER_TASK:=}"

# Output dirs — keep a tag so reruns with different CFG don't collide.
TAG="cfg${CFG}_a${ADAPTER_SCALE}"
A_DIR="$OUT/emmdit_pnp__$TAG"
B_DIR="$OUT/text_pnp__$TAG"
CSV="$OUT/compare_editclip_vs_text__$TAG.csv"

# ---------- helpers ----------
extra_args=()
if [[ -n "$CATEGORIES" ]]; then
  extra_args+=(--categories $CATEGORIES)
fi
if [[ -n "$TASKS" ]]; then
  extra_args+=(--tasks $TASKS)
fi
if [[ -n "$MAX_TESTS_PER_TASK" ]]; then
  extra_args+=(--max_tests_per_task "$MAX_TESTS_PER_TASK")
fi

echo "============================================================"
echo " Compare EditCLIP+E-MMDiT+PnP vs Llama+E-MMDiT+PnP"
echo "------------------------------------------------------------"
echo "  cfg=$CFG  adapter_scale=$ADAPTER_SCALE  k_ex=$K_EXEMPLARS"
echo "  PnP layers [$PNP_LAYER_START,$PNP_LAYER_END)  steps [$PNP_START_STEP,$PNP_STOP_STEP)"
echo "  steps inv=$NUM_INVERSION_STEPS  denoise=$NUM_DENOISING_STEPS  order=$SOLVER_ORDER"
echo "  A out -> $A_DIR"
echo "  B out -> $B_DIR"
echo "  CSV   -> $CSV"
echo "============================================================"

# ---------- Method A: EditCLIP + E-MMDiT + PnP ----------
echo
echo "[A] EditCLIP + E-MMDiT + PnP"
"$PY" topbench_eval/run_emmdit_pnp.py \
    --benchmark_root "$BENCH" \
    --finetuned_ckpt "$FINETUNED_CKPT" \
    --editclip_path  "$EDITCLIP_PATH" \
    --output_dir     "$A_DIR" \
    --clean_output_dir \
    --k_exemplars "$K_EXEMPLARS" \
    --cfg_scale "$CFG" \
    --adapter_scale "$ADAPTER_SCALE" \
    --pnp_layer_start "$PNP_LAYER_START" \
    --pnp_layer_end   "$PNP_LAYER_END" \
    --pnp_start_step "$PNP_START_STEP" \
    --pnp_stop_step  "$PNP_STOP_STEP" \
    --num_inversion_steps "$NUM_INVERSION_STEPS" \
    --num_denoising_steps "$NUM_DENOISING_STEPS" \
    --solver_order "$SOLVER_ORDER" \
    --resolution "$RESOLUTION" \
    --dtype "$DTYPE" \
    "${extra_args[@]}"

# ---------- Method B: Llama + E-MMDiT + PnP ----------
# Note: --adapter_scale is intentionally NOT passed here — run_text_pnp.py
# has no adapter, the prompt embeds come from Llama directly.
echo
echo "[B] Llama + E-MMDiT + PnP"
"$PY" topbench_eval/run_text_pnp.py \
    --benchmark_root "$BENCH" \
    --task_prompts   "$TASK_PROMPTS" \
    --output_dir     "$B_DIR" \
    --clean_output_dir \
    --cfg_scale "$CFG" \
    --pnp_layer_start "$PNP_LAYER_START" \
    --pnp_layer_end   "$PNP_LAYER_END" \
    --pnp_start_step "$PNP_START_STEP" \
    --pnp_stop_step  "$PNP_STOP_STEP" \
    --num_inversion_steps "$NUM_INVERSION_STEPS" \
    --num_denoising_steps "$NUM_DENOISING_STEPS" \
    --solver_order "$SOLVER_ORDER" \
    --resolution "$RESOLUTION" \
    --dtype "$DTYPE" \
    "${extra_args[@]}"

# ---------- Compare ----------
echo
echo "[compare] computing LPIPS / SSIM / EC2EC / CLIP Score..."
"$PY" topbench_eval/compare.py \
    --run "editclip+emmdit+pnp=$A_DIR" \
    --run "llama+emmdit+pnp=$B_DIR" \
    --editclip_path "$EDITCLIP_PATH" \
    --task_prompts  "$TASK_PROMPTS" \
    --output_csv    "$CSV" \
    --eval_size 256

echo
echo "Done."
echo "  CSV:  $CSV"
echo "  JSON: ${CSV%.csv}.json"
