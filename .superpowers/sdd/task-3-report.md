# Task 3 report: correctness-first packed W4A4 CUDA operator

## Status

Implemented the SM80 packed W4A4 operator, Python module, explicit CUDA/Meta
registrations, shape/device/dtype validation, and focused CUDA coverage.

## RED

Command:

```bash
source /venv/main/bin/activate
python -m pytest -q tests/test_linear_w4a4.py
```

Initial expected failure: import failed because `LinearW4A4` did not exist.
The first invocation used the system `pytest` and lacked torch; rerunning through
`python -m pytest` in `/venv/main` established the intended missing-interface RED.

## Build

The root build initially stopped in the vendored fast-hadamard-transform before
reaching FlatQuant because CUDA 13 rejects its SM70 target. A temporary local
vendor edit removed the SM70 pair, the dependency was installed, and the edit was
restored. No vendor source change is included in this commit.

Successful root build:

```bash
source /venv/main/bin/activate
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 python -m pip install -v -e . \
  --no-build-isolation --no-deps
```

Result: `deploy._CUDA` and `deploy._MARLIN` linked and the editable wheel installed.
Incremental verification builds used `MAX_JOBS=2 python setup.py build_ext --inplace`.

## GREEN

Final verification command:

```bash
source /venv/main/bin/activate
MAX_JOBS=2 python setup.py build_ext --inplace >/tmp/task3-final-build.log 2>&1 \
  && python -m pytest -q tests/test_linear_w4a4.py \
  && git diff --check
```

Result: `12 passed in 2.22s`; build and whitespace check exited zero.

The numeric debugging found that scalar CUDA `/ 7` produced a scale one ULP
different from PyTorch's elementwise CUDA reference on half-way cases. Computing
the scale with the rounded reciprocal matched PyTorch, while retaining true
division for activation quantization preserved ties-to-even behavior. The original
`rtol=2e-2, atol=2e-1` was not loosened.

## Changed files

- `deploy/nn/linear_w4a4.py`
- `deploy/nn/__init__.py`
- `deploy/kernels/w4a4_bindings.cpp`
- `deploy/kernels/w4a4_gemm.cu`
- `deploy/kernels/include/w4a4_gemm.h`
- `setup.py`
- `tests/test_linear_w4a4.py`

## Self-review

- Packing remains signed two's-complement, low nibble first, matching checkpoint format.
- CUTLASS uses signed int4 row-major A, column-major B, INT32 accumulation, SM80.
- Activation scales remain FP32 on device to avoid accuracy loss; weight scales are FP16.
- Scaling and clip consumption remain on device; no `Tensor.item()` host synchronization.
- Python wrapper preserves arbitrary leading batch dimensions and optional bias.
- CUDA and Meta registrations expose the required packed/scale/output shapes.
- Rejection tests cover architecture, BF16, alignment, scale shape, and batch shape.
- Generated CMake/build files and pre-existing dirty submodule state are not staged.

## Concerns

- The repository build still requires the known CUDA-13 workaround for the vendored
  fast-hadamard-transform on a clean environment. This task does not commit a vendor
  workaround because it is unrelated to the W4A4 operator and the brief explicitly
  permits a temporary adjustment.
- The root setup arch helper currently builds multiple supported architectures despite
  `TORCH_CUDA_ARCH_LIST=8.0`; the new operator rejects non-SM80 at runtime as required.
