#!/usr/bin/env bash
set -euo pipefail

WORKERS=16
BACKEND="gmr_baseline"
ROBOT="nao"
SOURCE_UP_AXIS="y"
SRC_ROOT=""
BEAT2_ROOT=""
MANIFEST="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv"
CONVERTED_ROOT="motion_data/BEAT2/converted"
SOURCE_CACHE_ROOT="motion_data/BEAT2/eval_cache/source"
SOURCE_FEATURE_DIR="motion_data/BEAT2/features/source"
SOURCE_ANOVA_DIR="motion_data/BEAT2/anova/source"
MODEL_ROOT="assets/body_models"
N_BOOTSTRAP=1000
SCALE_SAMPLE_LIMIT=0
FORCE_MANIFEST=0
SKIP_EXISTING=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/beat2_processing/run_backend_pipeline.sh [options]

Runs the full BEAT2 -> NAO backend pipeline:
  manifest -> retarget/source+robot caches -> Laban features -> ANOVA -> EFPR -> MPJPE/JJR/SCR

Options:
  --workers N
  --backend NAME
  --robot NAME
  --source_up_axis y|z
  --src_root PATH          Path to beat_english_v2.0.0/smplxflame_30.
  --beat2_root PATH        Path to BEAT2 root. Inferred from --src_root when omitted.
  --manifest PATH
  --converted_root PATH
  --source_cache_root PATH
  --source_feature_dir PATH
  --source_anova_dir PATH
  --model_root PATH
  --n_bootstrap N
  --scale_sample_limit N
  --force_manifest         Rebuild manifest even if it already exists.
  --skip_existing          Pass --skip_existing to batch_retarget_nao.py for partial resumes.
                           By default retarget/cache precompute overwrites outputs.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --robot)
      ROBOT="$2"
      shift 2
      ;;
    --source_up_axis)
      SOURCE_UP_AXIS="$2"
      shift 2
      ;;
    --src_root)
      SRC_ROOT="$2"
      shift 2
      ;;
    --beat2_root)
      BEAT2_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --converted_root)
      CONVERTED_ROOT="$2"
      shift 2
      ;;
    --source_cache_root)
      SOURCE_CACHE_ROOT="$2"
      shift 2
      ;;
    --source_feature_dir|--source_features_dir)
      SOURCE_FEATURE_DIR="$2"
      shift 2
      ;;
    --source_features)
      SOURCE_FEATURE_DIR="$(dirname "$2")"
      shift 2
      ;;
    --source_anova_dir)
      SOURCE_ANOVA_DIR="$2"
      shift 2
      ;;
    --model_root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --n_bootstrap)
      N_BOOTSTRAP="$2"
      shift 2
      ;;
    --scale_sample_limit)
      SCALE_SAMPLE_LIMIT="$2"
      shift 2
      ;;
    --force_manifest)
      FORCE_MANIFEST=1
      shift
      ;;
    --force_retarget)
      echo "[INFO] --force_retarget is deprecated: retarget/cache precompute runs by default."
      shift
      ;;
    --skip_existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RETARGETED_ROOT="motion_data/BEAT2/retargeted/${BACKEND}"
ROBOT_CACHE_ROOT="motion_data/BEAT2/eval_cache/${BACKEND}"
ROBOT_FEATURE_DIR="motion_data/BEAT2/features/${BACKEND}"
SOURCE_FEATURES="${SOURCE_FEATURE_DIR}/beat2_source_features.csv"
ROBOT_FEATURES="${ROBOT_FEATURE_DIR}/beat2_${ROBOT}_features.csv"
ROBOT_ANOVA_DIR="motion_data/BEAT2/anova/${BACKEND}"
EFPR_DIR="motion_data/BEAT2/efpr/${BACKEND}"
RETARGET_METRICS_DIR="motion_data/BEAT2/retarget_metrics/${BACKEND}"

manifest_dir="$(dirname "$MANIFEST")"
manifest_name="$(basename "$MANIFEST")"
if [[ "$manifest_name" != "beat2_emotion_manifest.csv" ]]; then
  echo "[ERROR] build_emotion_manifest.py always writes beat2_emotion_manifest.csv." >&2
  echo "        Use --manifest <dir>/beat2_emotion_manifest.csv or provide an existing custom manifest." >&2
  exit 2
fi

if [[ -z "$BEAT2_ROOT" && -n "$SRC_ROOT" ]]; then
  src_parent="$(cd "$SRC_ROOT/.." && pwd)"
  BEAT2_ROOT="$(cd "$src_parent/.." && pwd)"
fi

SRC_ROOT_ARGS=()
if [[ -n "$SRC_ROOT" ]]; then
  SRC_ROOT_ARGS=(--src_root "$SRC_ROOT")
fi

SKIP_EXISTING_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  SKIP_EXISTING_ARGS=(--skip_existing)
fi

manifest_rows() {
  if [[ ! -f "$MANIFEST" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "$MANIFEST")"
  if [[ "$lines" -gt 0 ]]; then
    echo $((lines - 1))
  else
    echo 0
  fi
}

echo "[1/9] Ensure BEAT2 emotion manifest"
if [[ "$FORCE_MANIFEST" -eq 0 && -f "$MANIFEST" ]]; then
  echo "[SKIP] Manifest exists: $MANIFEST"
else
  if [[ -z "$BEAT2_ROOT" ]]; then
    echo "[ERROR] Manifest is missing and BEAT2 root is unknown." >&2
    echo "        Pass --beat2_root /path/to/BEAT2 or --src_root /path/to/beat_english_v2.0.0/smplxflame_30." >&2
    exit 2
  fi
  echo "[RUN] Build manifest from: $BEAT2_ROOT"
  python scripts/beat2_processing/build_emotion_manifest.py \
    --beat2_root "$BEAT2_ROOT" \
    --output_dir "$manifest_dir"
fi

expected_rows="$(manifest_rows)"
if [[ "$expected_rows" -le 0 ]]; then
  echo "[ERROR] Manifest has no data rows: $MANIFEST" >&2
  exit 1
fi

echo "[2/9] Retarget and build source/robot caches for ${BACKEND}"
echo "[RUN] Precompute runs by default so retargeting code changes are reflected."
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers "$WORKERS" \
  --backend "$BACKEND" \
  --robot "$ROBOT" \
  --source_up_axis "$SOURCE_UP_AXIS" \
  --manifest "$MANIFEST" \
  "${SRC_ROOT_ARGS[@]}" \
  --converted_root "$CONVERTED_ROOT" \
  --source_cache_root "$SOURCE_CACHE_ROOT" \
  --retargeted_root "$RETARGETED_ROOT" \
  --robot_cache_root "$ROBOT_CACHE_ROOT" \
  --model_root "$MODEL_ROOT" \
  "${SKIP_EXISTING_ARGS[@]}"

echo "[3/9] Extract source-side Laban features"
python scripts/beat2_processing/extract_source_laban_features.py \
  --workers "$WORKERS" \
  --manifest "$MANIFEST" \
  --cache_root "$SOURCE_CACHE_ROOT" \
  --output_dir "$SOURCE_FEATURE_DIR"

echo "[4/9] Extract robot-side Laban features for ${BACKEND}"
python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers "$WORKERS" \
  --manifest "$MANIFEST" \
  --robot "$ROBOT" \
  --cache_root "$ROBOT_CACHE_ROOT" \
  --output_dir "$ROBOT_FEATURE_DIR"

echo "[5/9] Run source-side ANOVA"
python scripts/beat2_processing/run_anova.py \
  --features_csv "$SOURCE_FEATURES" \
  --output_dir "$SOURCE_ANOVA_DIR"

echo "[6/9] Run robot-side ANOVA for ${BACKEND}"
python scripts/beat2_processing/run_anova.py \
  --features_csv "$ROBOT_FEATURES" \
  --output_dir "$ROBOT_ANOVA_DIR"

echo "[7/9] Compute EFPR for ${BACKEND}"
python scripts/beat2_processing/compute_efpr.py \
  --human_anova "${SOURCE_ANOVA_DIR}/anova_main_table.csv" \
  --robot_anova "${ROBOT_ANOVA_DIR}/anova_main_table.csv" \
  --output_dir "$EFPR_DIR"

echo "[8/9] Compute bootstrap EFPR CI for ${BACKEND}"
python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features "$SOURCE_FEATURES" \
  --robot_features "$ROBOT_FEATURES" \
  --output_dir "$EFPR_DIR" \
  --n_bootstrap "$N_BOOTSTRAP"

echo "[9/9] Evaluate MPJPE / JJR / SCR for ${BACKEND}"
python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --workers "$WORKERS" \
  --robot "$ROBOT" \
  --manifest "$MANIFEST" \
  --source_cache_root "$SOURCE_CACHE_ROOT" \
  --robot_cache_root "$ROBOT_CACHE_ROOT" \
  --output_dir "$RETARGET_METRICS_DIR" \
  --scale_sample_limit "$SCALE_SAMPLE_LIMIT"

echo "[DONE] Full BEAT2 backend pipeline completed: ${BACKEND}"
