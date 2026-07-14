"""Native FlatQuant W4A4 operator dispatch."""

from collections import Counter
from threading import Lock

import torch
import deploy._CUDA  # noqa: F401  # Registers torch.ops.flatquant in each worker.
from deploy import w4a4_fake as _w4a4_fake  # noqa: F401

from .transform import apply_transform


class DispatchCounters:
    """Small thread-safe counter collection for W4A4 dispatch observability."""

    def __init__(self):
        self._counts = Counter({name: 0 for name in self.NAMES})
        self._lock = Lock()

    NAMES = ("w4a4", "w4a16_fallback", "bf16_fallback")

    def increment(self, name: str) -> None:
        if name not in self.NAMES:
            raise ValueError(f"unknown dispatch counter {name!r}")
        with self._lock:
            self._counts[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> dict[str, int]:
        """Clear counts and return the previous snapshot (RPC-friendly)."""
        with self._lock:
            previous = dict(self._counts)
            self._counts.clear()
            self._counts.update({name: 0 for name in self.NAMES})
            return previous


dispatch_counters = DispatchCounters()


# Keep Python observability outside Dynamo's traced bytecode. The marker has a
# declared mutation solely to keep it live in the compiled graph; it does not
# alter the activation and performs no CUDA allocation or synchronization.
@torch.library.custom_op("flatquant::record_w4a4_dispatch", mutates_args={"x"})
def _record_w4a4_dispatch(x: torch.Tensor) -> None:
    dispatch_counters.increment("w4a4")


@_record_w4a4_dispatch.register_fake
def _(x) -> None:
    return None


def apply_w4a4(layer, x, bias=None, *, prefix="<unknown>", policy=None,
               representations=("w4a4",)):
    leading_shape = x.shape[:-1]
    flat_x = x.reshape(-1, x.shape[-1])
    if policy is not None:
        if torch.compiler.is_compiling():
            # A Python comparison against the symbolic token count specializes
            # vLLM's dynamic compile range. Graph mode is valid only for the
            # artifact's sole implemented representation and all positive M.
            if policy.min_w4a4_rows != 1 or tuple(representations) != ("w4a4",):
                raise RuntimeError(
                    "CUDA Graph mode requires FLATQUANT_W4A4_MIN_ROWS=1 and a "
                    "W4A4-only artifact"
                )
        else:
            selected = policy.select(flat_x.shape[0], prefix, representations)
            if selected != "w4a4":
                raise RuntimeError(
                    f"fallback {selected!r} selected for prefix={prefix}, M={flat_x.shape[0]} "
                    "but no matching tensor load path is implemented"
                )
    transformed = apply_transform(layer, flat_x)
    packed_x, x_scale = torch.ops.flatquant.quantize_pack_i4(
        transformed.contiguous(), layer.activation_clip
    )
    output = torch.ops.flatquant.w4a4_linear(
        packed_x, layer.weight, x_scale, layer.weight_scale, x.dtype
    )
    output = output.reshape(*leading_shape, output.shape[-1])
    torch.ops.flatquant.record_w4a4_dispatch(output)
    return output if bias is None else output + bias
