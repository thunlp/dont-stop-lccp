"""Fail closed when a historical native-sink evaluation would silently use another kernel."""

import inspect
import sys


def require_native_sink(model):
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    if any("learnable_sink" not in inspect.signature(function).parameters
           for function in [flash_attn_func, flash_attn_varlen_func]):
        raise RuntimeError("Historical evaluation requires native FlashAttention learnable_sink; Triton fallback is not accepted")
    layers = [layer for layer in model.modules() if type(layer).__name__ == "MiniCPMFlashAttention2"]
    if not layers or any(not layer._should_use_sink() for layer in layers):
        raise RuntimeError("Expected active sink-enabled MiniCPM FlashAttention layers")
    module = sys.modules[type(layers[0]).__module__]
    if not getattr(module, "SUPPORT_FLASH_ATTN_SINK", False):
        raise RuntimeError("Model did not activate native sink dispatch")
    return {"backend": "native-flash-attention-sink", "layers": len(layers)}
