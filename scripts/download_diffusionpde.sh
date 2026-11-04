#!/usr/bin/env bash
# Download steady-state datasets (train + test splits) from the Hugging Face dataset
# jcy20/DiffusionPDE-normalized.
#
# Usage: scripts/download_diffusionpde.sh <output_dir> <dataset> [<dataset> ...]
#   e.g. scripts/download_diffusionpde.sh data/diffusionpde darcy helmholtz ns-nonbounded
#
# Requires the Hugging Face CLI (pip install -U huggingface-hub) and `hf auth login`.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

REPO_ID="jcy20/DiffusionPDE-normalized"
OUTPUT_DIR="$1"; shift
FOLDERS=()
for dataset in "$@"; do
  FOLDERS+=("${dataset}_hf" "${dataset}_test_hf")
done

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' not found. Install it with: pip install -U huggingface-hub" >&2
  exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in to Hugging Face. Run 'hf auth login' and, if the dataset is gated," >&2
  echo "accept its terms at https://huggingface.co/datasets/${REPO_ID}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
for folder in "${FOLDERS[@]}"; do
  echo "=== ${folder} ==="
  hf download "$REPO_ID" --repo-type dataset --include "${folder}/*" --local-dir "$OUTPUT_DIR"
  if [[ -z "$(ls -A "${OUTPUT_DIR}/${folder}" 2>/dev/null)" ]]; then
    echo "ERROR: ${OUTPUT_DIR}/${folder} is missing or empty" >&2
    exit 1
  fi
done
echo "Done: ${FOLDERS[*]} in ${OUTPUT_DIR}"
