#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VLLM_VENV_PATH:-/workspace/.venvs/flatquant-vllm}"
PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
uv pip install -r "${REPO_ROOT}/requirements-vllm.txt"
uv pip install -e "${REPO_ROOT}/vllm_plugin" --no-deps

python - <<'PY'
import torch
import triton
import vllm

print(f"vLLM {vllm.__version__}")
print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
print(f"Triton {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
