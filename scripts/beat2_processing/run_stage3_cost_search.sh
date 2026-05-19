#!/usr/bin/env bash
set -euo pipefail

COSTS=(1 5 10)
COMMON_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/beat2_processing/run_stage3_cost_search.sh [options] [pipeline options]

Runs the full BEAT2 -> NAO pipeline for multiple gmr_velocity_stage3_wrist costs.

Default costs:
  1 5 10

Each run uses:
  --backend gmr_velocity_stage3_wrist
  --output_backend gmr_velocity_stage3_wrist_<cost>
  --velocity_stage3_cost <cost>

Examples:
  bash scripts/beat2_processing/run_stage3_cost_search.sh \
    --workers 16 \
    --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30 \
    --beat2_root /path/to/BEAT2

  bash scripts/beat2_processing/run_stage3_cost_search.sh \
    --costs "1 3 5 10" \
    --workers 16 \
    --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30

Options handled by this wrapper:
  --costs "A B C"        Space-separated stage3 costs. Defaults to "1 5 10 50 100".
  -h, --help

All other options are forwarded to run_backend_pipeline.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --costs)
      read -r -a COSTS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#COSTS[@]}" -eq 0 ]]; then
  echo "[ERROR] --costs cannot be empty." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for cost in "${COSTS[@]}"; do
  label="${cost//./p}"
  label="${label//-/m}"
  output_backend="gmr_velocity_stage3_wrist_${label}"

  echo
  echo "================================================================"
  echo "[RUN] stage3 cost=${cost} -> output_backend=${output_backend}"
  echo "================================================================"

  bash scripts/beat2_processing/run_backend_pipeline.sh \
    --backend gmr_velocity_stage3_wrist \
    --output_backend "$output_backend" \
    --velocity_stage3_cost "$cost" \
    "${COMMON_ARGS[@]}"
done

echo
echo "[DONE] Stage3 cost search completed for costs: ${COSTS[*]}"
