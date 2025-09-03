#!/usr/bin/env bash
set -euo pipefail

# Convert VERL FSDP SFT checkpoints to HuggingFace format
# Either provide -l LOCAL_DIR (path to .../global_step_*/actor) or -o OUTDIR (root from training)
# Required: -t TARGET_DIR where the HF model will be written

usage(){
  echo "Usage: $0 [-l LOCAL_DIR | -o OUTDIR] -t TARGET_DIR" >&2
  echo "  LOCAL_DIR: path to the 'actor' directory, e.g., /.../global_step_final/actor" >&2
  echo "  OUTDIR: training output directory; we'll default to OUTDIR/global_step_final/actor" >&2
}

LOCAL_DIR=""
OUTDIR=""
TARGET_DIR=""

while getopts ":l:o:t:h" opt; do
  case ${opt} in
    l) LOCAL_DIR="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    t) TARGET_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 1 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TARGET_DIR" ]]; then
  echo "Error: -t TARGET_DIR is required" >&2
  usage
  exit 1
fi

# Resolve absolute paths
abspath(){ python3 - "$1" <<'PY'
import os,sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

TARGET_DIR=$(abspath "$TARGET_DIR")
if [[ -n "$OUTDIR" && -z "$LOCAL_DIR" ]]; then
  OUTDIR=$(abspath "$OUTDIR")
  LOCAL_DIR="$OUTDIR/global_step_final/actor"
fi

if [[ -z "$LOCAL_DIR" ]]; then
  echo "Error: Provide either -l LOCAL_DIR or -o OUTDIR" >&2
  usage
  exit 1
fi

LOCAL_DIR=$(abspath "$LOCAL_DIR")

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Error: LOCAL_DIR does not exist: $LOCAL_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

VERL_DIR="/home/nickatomlin/georgiazhou/self_play/verl"
export PYTHONPATH="$VERL_DIR:$PYTHONPATH"

echo "Merging FSDP checkpoint from: $LOCAL_DIR"
echo "Saving HuggingFace model to: $TARGET_DIR"

exec python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR" 