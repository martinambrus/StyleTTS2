from monotonic_align import maximum_path
from monotonic_align.core import maximum_path_c
import math
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
def log_norm(x, mean=-4, std=4, dim=2):
    """Compute a numerically-stable log magnitude norm.

    Training can push the latent magnitudes feeding this helper well outside
    the range that single-precision arithmetic can represent.  The previous
    fixes avoided overflow in typical scenarios but still allowed extremely
    large activations to turn into ``inf``/``nan`` which then poisoned the
    gradients for the rest of the epoch.  To make the routine fully robust we
    perform the reduction in float64, sanitise any non-finite inputs with
    :func:`torch.nan_to_num`, and clamp the intermediate results to the largest
    representable magnitudes for the working dtype.  This sacrifices no
    precision in the usual operating regime while ensuring the function always
    returns finite values, even for adversarially large activations or entirely
    padded regions.
    """

    scaled = x * std + mean

    working = scaled.to(torch.float64)
    finfo = torch.finfo(working.dtype)
    # When ``working`` is extremely large we still want ``2 * working`` to be
    # within the representable range of ``logsumexp``.  The log of the maximum
    # finite value gives us that boundary.
    clamp_bound = 0.5 * math.log(finfo.max)
    working = torch.nan_to_num(working, nan=0.0, posinf=clamp_bound, neginf=-clamp_bound)
    working = torch.clamp(working, min=-clamp_bound, max=clamp_bound)

    log_sum = torch.logsumexp(2 * working, dim=dim)

    min_log_value = math.log(finfo.tiny)
    max_log_value = math.log(finfo.max)
    log_sum = torch.clamp(log_sum, min=min_log_value, max=max_log_value)

    return 0.5 * log_sum.to(scaled.dtype)

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
    
