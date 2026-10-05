#!/usr/bin/env bash
set -euo pipefail

# Usage: launch.sh USER_ROOT MODEL DATA_DIR OUTPUT [--prepare-only|--resume]
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 USER_ROOT MODEL DATA_DIR OUTPUT [--prepare-only|--resume]" >&2
  exit 2
fi
user_root=$1
model=$2
data_dir=$3
output=$4
shift 4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_python=${RUNTIME_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 PYTHONPATH="${script_dir}/.."
export TMPDIR="${output}/runtime/tmp" TMP="${output}/runtime/tmp" TEMP="${output}/runtime/tmp"
export XDG_CACHE_HOME="${output}/runtime/cache" HF_HOME="${output}/runtime/cache/hf"
export TORCH_HOME="${output}/runtime/cache/torch" TORCHINDUCTOR_CACHE_DIR="${output}/runtime/cache/torchinductor"
export TRITON_CACHE_DIR="${output}/runtime/cache/triton" CUDA_CACHE_PATH="${output}/runtime/cache/cuda"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME"
exec "$runtime_python" -B -u "${script_dir}/train.py" \
  --config "${script_dir}/v12.yaml" --gpus 4 --nodes 1 \
  --user-root "$user_root" --model "$model" --data-dir "$data_dir" --output "$output" "$@"
