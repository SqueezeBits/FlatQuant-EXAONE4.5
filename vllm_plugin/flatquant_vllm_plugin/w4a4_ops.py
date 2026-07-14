"""Native FlatQuant W4A4 operator dispatch."""

from collections import Counter
from threading import Lock

import torch
import deploy._CUDA  # noqa: F401  # Registers torch.ops.flatquant in each worker.

from .transform import apply_transform


class DispatchCounters:
    """Small thread-safe counter collection for W4A4 dispatch observability."""

    def __init__(self):
        self._counts = Counter()
        self._lock = Lock()

    def increment(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


dispatch_counters = DispatchCounters()


def apply_w4a4(layer, x, bias=None):
    transformed = apply_transform(layer, x)
    packed_x, x_scale = torch.ops.flatquant.quantize_pack_i4(
        transformed.contiguous(), layer.activation_clip
    )
    output = torch.ops.flatquant.w4a4_linear(
        packed_x, layer.weight, x_scale, layer.weight_scale, x.dtype
    )
    dispatch_counters.increment("w4a4")
    return output if bias is None else output + bias
