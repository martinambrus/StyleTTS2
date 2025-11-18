from monotonic_align import maximum_path
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import matplotlib.pyplot as plt
from munch import Munch


_SUPPORTED_MIXED_PRECISION = {"no", "fp16", "bf16"}

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2, clamp=(-60.0, 60.0), eps=1e-20):
    """Compute ``log ||exp(x * std + mean)||`` with aggressive sanitisation.

    The helper guards the norm-consistency loss against pathological inputs by
    (1) moving the computation into float64, (2) replacing non-finite values,
    (3) applying a log-sum-exp formulation centred around the maximum element to
    avoid overflow, and (4) clamping the final activations to a conservative
    range that still covers typical speech magnitudes.  These safeguards ensure
    that the downstream predictor never receives ``nan``/``inf`` targets while
    keeping gradients well behaved.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("log_norm expects a torch.Tensor input")

    scaled = x * std + mean
    # Replace NaNs/Infs before promoting the dtype to avoid propagating them.
    scaled = torch.nan_to_num(scaled, nan=mean, posinf=mean + 10 * std, neginf=mean - 10 * std)

    working = scaled.to(torch.float64)
    if working.ndim == 0:
        working = working.unsqueeze(0)

    dim = dim if dim >= 0 else working.ndim + dim

    finfo = torch.finfo(working.dtype)

    finite_mask = torch.isfinite(working)
    if not finite_mask.all():
        working = torch.where(finite_mask, working, working.new_full((), mean))

    max_val, _ = working.max(dim=dim, keepdim=True)
    max_val = torch.where(torch.isfinite(max_val), max_val, working.new_full(max_val.shape, mean))

    centered = working - max_val
    sum_exp = torch.exp(2 * centered).sum(dim=dim)
    sum_exp = torch.nan_to_num(sum_exp, nan=0.0, posinf=finfo.max)

    log_values = max_val.squeeze(dim) + 0.5 * torch.log(sum_exp + eps)
    log_values = torch.nan_to_num(log_values, nan=mean, posinf=finfo.max, neginf=-finfo.max)

    if clamp is not None:
        clamp_min, clamp_max = clamp
        if clamp_min is not None:
            log_values = torch.maximum(log_values, log_values.new_tensor(clamp_min))
        if clamp_max is not None:
            log_values = torch.minimum(log_values, log_values.new_tensor(clamp_max))

    return log_values.to(scaled.dtype)


def ensure_finite(tensor, *, clamp=None, fill_value=0.0):
    """Replace non-finite values in ``tensor`` and optionally clamp its range."""

    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor

    if not tensor.is_floating_point():
        return tensor

    finfo = torch.finfo(tensor.dtype)
    if clamp is None:
        min_val, max_val = -finfo.max, finfo.max
    else:
        min_val, max_val = clamp
        if min_val is None:
            min_val = -finfo.max
        if max_val is None:
            max_val = finfo.max

    sanitized = torch.nan_to_num(tensor, nan=fill_value, posinf=max_val, neginf=min_val)
    sanitized = sanitized.clamp(min=min_val, max=max_val)
    return sanitized

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)


def _get_device_index(device_index=None):
    """Return a valid CUDA device index or ``None`` if CUDA is unavailable."""

    if not torch.cuda.is_available():
        return None
    if device_index is None:
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0
    return device_index


def _cuda_device_capability(device_index=None):
    """Safely query the CUDA capability for the requested device."""

    index = _get_device_index(device_index)
    if index is None:
        return None
    try:
        return torch.cuda.get_device_capability(index)
    except Exception:
        return None


def _cuda_device_name(device_index=None):
    """Fetch the CUDA device name if available."""

    index = _get_device_index(device_index)
    if index is None:
        return ""
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def _supports_bfloat16(device_name, capability):
    """Determine whether the device can execute bfloat16 kernels efficiently."""

    if capability is None:
        return False

    major, minor = capability

    if major > 8:
        # Hopper (H100/H200) and newer architectures (e.g. Blackwell B-series)
        # expose native bfloat16 tensor core support.
        return True

    if major == 8 and minor == 0:
        # Ampere data-center GPUs (A100) also provide bfloat16 tensor cores.
        return True

    normalized_name = device_name.upper()
    return any(token in normalized_name for token in ("H100", "H200", "B100", "B200"))


def select_accelerate_mixed_precision(preference="auto", device_index=None):
    """Resolve the mixed precision mode to pass to :class:`Accelerator`.

    The helper keeps existing behaviour when users explicitly specify a mode,
    while mapping ``"auto"`` to ``"bf16"`` on GPUs that are known to support
    bfloat16 tensor cores (A100, H100/H200, B100/B200, and newer) and falling
    back to ``"fp16"`` otherwise.  When CUDA is not available, ``"no"`` is
    returned so the caller can disable mixed precision altogether.
    """

    if preference is None:
        preference = "auto"

    preference = str(preference).strip().lower()

    if preference != "auto":
        if preference not in _SUPPORTED_MIXED_PRECISION:
            raise ValueError(
                f"Unsupported mixed precision mode '{preference}'."
                f" Expected one of {_SUPPORTED_MIXED_PRECISION | {'auto'}}."
            )
        return preference

    capability = _cuda_device_capability(device_index)
    device_name = _cuda_device_name(device_index)

    if capability is None:
        return "no"

    if _supports_bfloat16(device_name, capability):
        return "bf16"

    # Default to fp16 on CUDA devices without bfloat16 tensor cores.
    return "fp16"


def describe_cuda_device(device_index=None):
    """Human readable representation of the selected CUDA device."""

    if not torch.cuda.is_available():
        return "CPU"

    index = _get_device_index(device_index)
    name = _cuda_device_name(index)
    capability = _cuda_device_capability(index)

    if capability is None:
        return name or "Unknown CUDA device"

    major, minor = capability
    return f"{name} (compute capability {major}.{minor})"
    
