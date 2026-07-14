#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 LABEL MODEL_PATH [vllm_awq.py latency arguments...]" >&2
    exit 2
fi

label=$1
model_path=$2
shift 2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
output_dir="${repo_root}/outputs/benchmark_results/profiles"
mkdir -p "${output_dir}"

exec env PYTHONPATH="${repo_root}/vllm_plugin" \
    nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --output="${output_dir}/${label}" \
    /workspace/.venv/bin/python \
    "${script_dir}/vllm_awq.py" latency \
    --model_path "${model_path}" \
    "$@"
