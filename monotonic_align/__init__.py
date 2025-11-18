"""Pure Python fallbacks for the ``monotonic_align`` package.

The original project ships a set of Cython/CUDA extensions that are
frequently recompiled during installation.  The compilation step fails on
newer NVIDIA architectures (e.g. Hopper/Blackwell) when the bundled NVRTC
version cannot recognise the automatically selected ``--gpu-architecture``.
To keep StyleTTS2 trainable on such GPUs we provide a fully self-contained
implementation of the two helpers that the project relies on: the
``mask_from_lens`` utility and the dynamic-programming ``maximum_path``
search.

The algorithms mirror the behaviour of
``https://github.com/resemble-ai/monotonic_align`` but run entirely on the
CPU using NumPy/Torch tensors, removing the dependency on external build
toolchains.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["mask_from_len", "mask_from_lens", "maximum_path"]


def mask_from_len(lens: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Return a boolean mask that is ``True`` for valid sequence positions."""

    if not isinstance(lens, torch.Tensor):
        raise TypeError("lens must be a torch.Tensor")

    if max_len is None:
        if lens.numel() == 0:
            max_len = 0
        else:
            max_len = int(lens.max().item())

    if max_len < 0:
        raise ValueError("max_len must be non-negative")

    index = torch.arange(max_len, device=lens.device)
    return index.view(1, -1) < lens.unsqueeze(1)


def mask_from_lens(
    similarity: torch.Tensor,
    symbol_lens: torch.Tensor,
    mel_lens: torch.Tensor,
) -> torch.Tensor:
    """Replicate the helper from ``monotonic_align.mas`` for compatibility."""

    if similarity.ndim != 3:
        raise ValueError("similarity tensor must be 3-D (B, S, T)")

    _, symbols, mels = similarity.size()
    mask_s = mask_from_len(symbol_lens, symbols)
    mask_t = mask_from_len(mel_lens, mels)
    return (mask_s.unsqueeze(2) * mask_t.unsqueeze(1)).to(similarity.dtype)


def _maximum_path_each(
    path: np.ndarray,
    value: np.ndarray,
    t_x: int,
    t_y: int,
    max_neg_val: float,
) -> None:
    """In-place dynamic programming kernel operating on NumPy arrays."""

    if t_x <= 0 or t_y <= 0:
        return

    max_neg = float(max_neg_val)

    for y in range(t_y):
        x_start = max(0, t_x + y - t_y)
        x_end = min(t_x, y + 1)
        for x in range(x_start, x_end):
            if x == y:
                v_cur = max_neg
            else:
                v_cur = value[x, y - 1]

            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg
            else:
                v_prev = value[x - 1, y - 1]

            value[x, y] = max(v_cur, v_prev) + value[x, y]

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index == 0:
            continue

        take_diag = index == y
        if not take_diag and y > 0:
            take_diag = value[index, y - 1] < value[index - 1, y - 1]
        if take_diag:
            index -= 1


def maximum_path(
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    max_neg_val: float = -1e9,
) -> torch.Tensor:
    """Return the most likely monotonic alignment path for ``value``.

    The function mirrors ``monotonic_align.mas.maximum_path`` but only
    implements the 1-step topology that StyleTTS2 uses.  The work is carried
    out on CPU NumPy arrays which avoids triggering NVRTC/NVCC.
    """

    if value.ndim != 3:
        raise ValueError("value tensor must be 3-D (B, text, mel)")

    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    elif mask.shape != value.shape:
        raise ValueError("mask must have the same shape as value")

    device = value.device
    dtype = value.dtype

    value_np = value.detach().to(torch.float32).cpu().numpy().copy()
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)

    paths = np.zeros_like(value_np, dtype=np.int32)
    t_x_max = mask_np.sum(axis=1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(axis=2)[:, 0].astype(np.int32)

    for batch in range(value_np.shape[0]):
        _maximum_path_each(
            paths[batch],
            value_np[batch],
            int(t_x_max[batch]),
            int(t_y_max[batch]),
            max_neg_val,
        )

    return torch.from_numpy(paths).to(device=device, dtype=dtype)

