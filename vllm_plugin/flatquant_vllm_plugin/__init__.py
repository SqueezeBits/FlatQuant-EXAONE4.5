"""FlatQuant W4A16 quantization plugin for vLLM."""


def register():
    # Importing config executes the decorator in every vLLM process.
    from . import config  # noqa: F401
