"""FlatQuant W4A16 and W4A4 quantization plugins for vLLM."""


def register():
    # Importing configs executes the decorators in every vLLM process.
    from . import config  # noqa: F401
    from . import w4a4_config  # noqa: F401
