"""Utilities to keep training scripts compatible with GPUs lacking NVRTC support.

This module is imported before :mod:`torch` so that we can force PyTorch to use
pre-built CUDA kernels instead of the NVRTC JIT (nvFuser, TorchInductor, etc.).
The NVRTC toolchain bundled with many PyTorch wheels does not yet recognise the
latest GPU architectures (e.g. NVIDIA B-series).  Attempting to compile a
kernel on those GPUs fails with ``nvrtc: error: invalid value for
--gpu-architecture (-arch)`` and aborts training.

Importing this module sets a handful of conservative environment defaults that
turn off runtime compilation.  After :mod:`torch` is imported the helper can be
called once more to disable any remaining nvFuser hooks.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Tuple

__all__ = ["disable_cuda_jit_features_if_available"]


_CUDA_JIT_ENV_DEFAULTS = {
    # Force PyTorch to skip TorchInductor/torch.compile paths entirely.
    "TORCHINDUCTOR_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    # Disable the nvFuser backend and keep the legacy CUDA fuser selected.
    "PYTORCH_NVFUSER_DISABLE": "1",
    "PYTORCH_JIT_ENABLE_NVFUSER": "0",
    "TORCH_CUDA_FUSER": "old",
}


def _set_default_env(overrides: Iterable[Tuple[str, str]]) -> None:
    for key, value in overrides:
        if key in os.environ:
            continue
        os.environ[key] = value


_set_default_env(_CUDA_JIT_ENV_DEFAULTS.items())


def disable_cuda_jit_features_if_available() -> bool:
    """Best-effort attempt to disable nvFuser hooks after :mod:`torch` import.

    Returns ``True`` if at least one runtime compiler hook was disabled.
    ``False`` is returned when :mod:`torch` is unavailable or if none of the
    expected internal helpers are present (older versions of PyTorch).
    """

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure is rare
        warnings.warn(
            f"Unable to import torch while disabling CUDA JIT features: {exc}"
        )
        return False

    disabled_any = False
    torch_compiler_guards = [
        ("_jit_set_nvfuser_enabled", (False,)),
        ("_jit_set_texpr_fuser_enabled", (False,)),
        ("_jit_override_can_fuse_on_gpu", (False,)),
        ("_jit_override_can_fuse_on_cpu", (False,)),
    ]

    for attr_name, args in torch_compiler_guards:
        fn = getattr(torch._C, attr_name, None)  # type: ignore[attr-defined]
        if fn is None:
            continue
        try:
            fn(*args)
            disabled_any = True
        except Exception:  # pragma: no cover - safety net for older versions
            continue

    return disabled_any
