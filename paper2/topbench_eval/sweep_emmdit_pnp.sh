#!/usr/bin/env bash
# Sweep over (cfg_scale, adapter_scale) for the EditCLIP + E-MMDiT + PnP method.
# Each (cfg, adapter) combo writes to a distinct output dir, so nothing
# collides with prior runs. All runs use --clean_output_dir for safety.
#
# Usage:
#   bash topbench_eval/sweep_emmdit_pnp.sh
#
# Override defaults by setting env vars before invocation, e.g.:
#   FINETUNED_CKPT=... EDITCLIP_PATH=... BENCH=... OUT=... \
#     bash topbench_eval/sweep_emmdit_pnp.sh

set -euo pipefail

# ---------- defaults ----------
: "${PY:=/root/e_mmdit/venv/bin/python}"
: "${BENCH:=/root/e_mmdit/flash-attention/Nitro-E-main/benchmark}"
: "${OUT:=/root/e_mmdit/flash-attention/Nitro-E-main/output/topbench}"
: "${FINETUNED_CKPT:=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_emmdit_mlp_adapter/2026_05_14-01_10_06_lr5e-05/checkpoint-102000/transformer.pt}"
: "${EDITCLIP_PATH:=/root/e_mmdit/flash-attention/Nitro-E-main/output/editclip_omnistyle_resumed/2026_05_13-04_01_34/checkpoint-100000}"
: "${PNP_LAYER_START:=4}"
: "${PNP_LAYER_END:=20}"
: "${K_EXEMPLARS:=1}"
: "${TASKS_FILTER:=}"       # e.g. "boy2girl charcoal apple" — leave empty for all
: "${CATEGORIES_FILTER:=}"  # e.g. "01"

# Sweep grid. Add/remove pairs as needed.
GRID=(
  "1.5 0.7"
  "2.5 0.7"   # mild edit + slightly weaker adapter — usually the sweet spot
  "2.5 1.0"   # standard adapter, mild CFG
  "3.5 1.0"
  "4.5 1.0"
)

# ---------- helpers ----------
extra_args=()
if [[ -n "$TASKS_FILTER" ]]; then
  extra_args+=(--tasks $TASKS_FILTER)
fi
if [[ -n "$CATEGORIES_FILTER" ]]; then
  extra_args+=(--categories $CATEGORIES_FILTER)
fi

# ---------- sweep ----------
for combo in "${GRID[@]}"; do
  read -r CFG AS <<< "$combo"
  TAG="cfg${CFG}_a${AS}"
  ODIR="$OUT/emmdit_pnp__$TAG"
  echo
  echo "======================================================"
  echo "[sweep] cfg=$CFG  adapter_scale=$AS  ->  $ODIR"
  echo "======================================================"
  "$PY" topbench_eval/run_emmdit_pnp.py \
      --benchmark_root "$BENCH" \
      --finetuned_ckpt "$FINETUNED_CKPT" \
      --editclip_path  "$EDITCLIP_PATH" \
      --output_dir     "$ODIR" \
      --clean_output_dir \
      --k_exemplars "$K_EXEMPLARS" \
      --cfg_scale "$CFG" \
      --adapter_scale "$AS" \
      --pnp_layer_start "$PNP_LAYER_START" \
      --pnp_layer_end   "$PNP_LAYER_END" \
      --pnp_start_step 0 --pnp_stop_step 30 \
      --solver_order 2 --num_inversion_steps 30 --num_denoising_steps 30 \
      --resolution 512 --dtype bf16 \
      "${extra_args[@]}"
done

echo
echo "[sweep] all combos done. To produce a comparison CSV:"
echo "  $PY topbench_eval/compare.py \\"
for combo in "${GRID[@]}"; do
  read -r CFG AS <<< "$combo"
  TAG="cfg${CFG}_a${AS}"
  echo "      --run \"emmdit_pnp_$TAG=$OUT/emmdit_pnp__$TAG\" \\"
done
echo "      --editclip_path $EDITCLIP_PATH \\"
echo "      --task_prompts topbench_eval/task_prompts.json \\"
echo "      --output_csv $OUT/sweep_emmdit_pnp.csv"
