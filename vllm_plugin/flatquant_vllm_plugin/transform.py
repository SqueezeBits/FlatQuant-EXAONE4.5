import math

from .triton_transform_v2 import kron_transform, left_transform


def decompose_dim(size):
    a = math.isqrt(size)
    if a * a < size:
        a += 1
    while True:
        delta = a * a - size
        b = math.isqrt(delta)
        if b * b == delta:
            return a - b, a + b
        a += 1


def apply_transform(layer, x):
    left = layer.flatquant_left
    right = getattr(layer, "flatquant_right", None)
    if right is None:
        return left_transform(x, left)
    return kron_transform(x, left, right)
