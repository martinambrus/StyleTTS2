#!/usr/bin/env python
"""Run the core AuxiliaryASR notebook diagnostics from the command line.

This helper consolidates the metrics from the ``AuxiliaryASR_*`` analysis
notebooks so they can be executed in one pass without launching Jupyter.  It
loads the latest checkpoint from the supplied directory, evaluates the
validation split, and prints notebook-style summaries for quick reporting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import math
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import yaml

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (  # noqa: E402
    build_dev_dataloader_from_config,
    load_asr_model_from_config,
    select_logits_from_output,
)


def _collect_epoch_checkpoints(path: Path) -> List[Tuple[int, Path]]:
    """Return sorted ``(epoch, path)`` tuples for ``epoch_*.pth`` files."""

    epoch_files: List[Tuple[int, Path]] = []
    if not path.is_dir():
        return epoch_files

    for file in path.glob("epoch_*.pth"):
        parts = file.stem.split("_")
        try:
            epoch = int(parts[-1])
        except (ValueError, IndexError):
            continue
        epoch_files.append((epoch, file))

    epoch_files.sort(key=lambda item: item[0])
    return epoch_files


def _evaluate_epoch_checkpoint(
    epoch_idx: int,
    checkpoint_path: str,
    config: Dict,
    device_str: str,
    band: float,
    max_align_batches: Optional[int],
) -> Dict[str, float]:
    """Evaluate a single epoch checkpoint and return its aggregated metrics."""

    config_copy = copy.deepcopy(config)
    device = torch.device(device_str)

    dev_loader, _ = build_dev_dataloader_from_config(config_copy, device)
    model, token_map = load_asr_model_from_config(config_copy, checkpoint_path, device)

    if " " not in token_map:
        raise KeyError("The vocabulary does not contain the blank symbol ' '.")
    blank_id = int(token_map[" "])

    entropy_stats = compute_ctc_entropy(model, dev_loader, device, blank_id)
    mean_diag, _ = diagonal_attention_score(
        model,
        dev_loader,
        device,
        band=band,
        max_batches=max_align_batches,
    )
    dur_stats = duration_stats(model, dev_loader, device, blank_id)
    skip_stats = skip_merge_flags(model, dev_loader, device, blank_id)
    per_stats = compute_phoneme_error_rate(model, dev_loader, device, blank_id)

    p50_for_penalty = (
        dur_stats["p50"] if math.isfinite(dur_stats["p50"]) else 0.0
    )
    joint_score = (
        per_stats["per"]
        + 0.25 * (1.0 - mean_diag)
        + 0.20 * max(0.0, entropy_stats["mean_blank_rate"] - 0.62)
        + 0.20 * max(0.0, (-skip_stats["mean_len_diff"] - 6.0) / 20.0)
        + 0.15 * max(0.0, 2.0 - p50_for_penalty)
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "joint_score": joint_score,
        "per": per_stats["per"],
        "diagonal": mean_diag,
        "blank_rate": entropy_stats["mean_blank_rate"],
        "len_diff": skip_stats["mean_len_diff"],
        "p50": dur_stats["p50"],
    }


def _resolve_checkpoint_path(path: Path) -> Path:
    """Return the checkpoint file that should be evaluated."""

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path '{path}' does not exist")

    preferred = [
        "best_joint.pth",
        "best_model.pth",
        "best_per.pth",
        "best_loss.pth",
    ]
    for name in preferred:
        candidate = path / name
        if candidate.exists():
            return candidate

    epoch_files = _collect_epoch_checkpoints(path)
    if not epoch_files:
        raise FileNotFoundError(
            f"Could not find any epoch checkpoints inside '{path}'."
        )

    return epoch_files[-1][1]


def _load_config(config_path: Optional[Path]) -> Dict:
    if config_path is None:
        raise FileNotFoundError("A configuration file must be provided")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _infer_config_path(checkpoint: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    if checkpoint.is_file():
        candidate = checkpoint.parent / "config.yml"
        if candidate.exists():
            return candidate
    else:
        candidate = checkpoint / "config.yml"
        if candidate.exists():
            return candidate
    fallback = ROOT_DIR / "Configs" / "config.yml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Unable to locate a configuration file. Pass --config explicitly."
    )


def _extract_ctc_logits(output) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("logits_ctc", "ctc_logits", "primary_logits", "logits"):
            tensor = output.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor
    if isinstance(output, (tuple, list)) and output:
        candidate = output[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    tensor = select_logits_from_output(output)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Unable to extract logits tensor from model output")
    return tensor


def _levenshtein_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    """Return the edit distance between two token sequences."""

    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    # Ensure that the dynamic programming buffer tracks the shorter sequence.
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, 1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + substitution_cost,  # substitution
                )
            )
        previous = current

    return previous[-1]


@torch.no_grad()
def compute_ctc_entropy(model: torch.nn.Module, dev_loader, device, blank_id: int) -> Dict[str, float]:
    model.eval()
    entropies: List[float] = []
    max_probs: List[float] = []
    blank_rates: List[float] = []
    downsample_factor = 2 ** getattr(model, "n_down", 1)

    for batch in dev_loader:
        texts, text_lens, mels, mel_lens = batch[:4]
        mels = mels.to(device)
        mel_lens = mel_lens.to(torch.long)

        outputs = model(mels)
        logits = _extract_ctc_logits(outputs)

        logp = logits.log_softmax(-1).cpu()
        probs = logp.exp()
        entropy = -(probs * logp).sum(-1)
        maxp, ids = probs.max(-1)
        blank_mask = ids.eq(blank_id)

        logit_lens = torch.clamp(
            mel_lens // downsample_factor,
            min=1,
            max=logits.size(1),
        ).cpu()

        for h, m, bmask, L in zip(entropy, maxp, blank_mask, logit_lens):
            length = int(L)
            entropies.append(h[:length].mean().item())
            max_probs.append(m[:length].mean().item())
            blank_rates.append(bmask[:length].float().mean().item())

    return {
        "mean_entropy": float(torch.tensor(entropies).mean()),
        "mean_maxprob": float(torch.tensor(max_probs).mean()),
        "mean_blank_rate": float(torch.tensor(blank_rates).mean()),
    }


@torch.no_grad()
def compute_phoneme_error_rate(
    model: torch.nn.Module,
    dev_loader,
    device: torch.device,
    blank_id: int,
) -> Dict[str, float]:
    """Return corpus-level phoneme error statistics using greedy CTC decoding."""

    model.eval()
    total_distance = 0
    total_reference = 0
    downsample_factor = 2 ** getattr(model, "n_down", 1)

    for batch in dev_loader:
        texts, text_lens, mels, mel_lens = batch[:4]
        mels = mels.to(device)
        mel_lens = mel_lens.to(torch.long)
        text_lens = text_lens.to(torch.long)

        outputs = model(mels)
        logits = _extract_ctc_logits(outputs).detach().cpu()
        logit_lens = torch.clamp(
            mel_lens // downsample_factor,
            min=1,
            max=logits.size(1),
        ).cpu()

        collapsed_ids, _ = best_path_durations(logits, logit_lens, blank_id)
        text_lens_list = text_lens.cpu().tolist()

        for pred_ids, target_tensor, tgt_len in zip(
            collapsed_ids,
            texts,
            text_lens_list,
        ):
            reference = target_tensor[: int(tgt_len)].tolist()
            distance = _levenshtein_distance(reference, pred_ids)
            total_distance += distance
            total_reference += len(reference)

    if total_reference == 0:
        raise RuntimeError("No reference tokens found while computing PER")

    per = float(total_distance) / float(total_reference)
    return {
        "per": per,
        "total_distance": float(total_distance),
        "total_reference": float(total_reference),
    }


@torch.no_grad()
def compute_ctc_loss(model: torch.nn.Module, dev_loader, device, blank_id: int) -> float:
    model_was_training = model.training
    model.eval()

    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    losses: List[float] = []
    downsample = 2 ** getattr(model, "n_down", 1)

    for batch in dev_loader:
        texts, text_lens, mels, mel_lens = batch[:4]

        mels = mels.to(device)
        targets = texts.to(device=device, dtype=torch.long)
        text_lens = text_lens.to(torch.long)
        mel_lens = mel_lens.to(torch.long)

        outputs = model(mels)
        logits = _extract_ctc_logits(outputs)

        if logits.dim() != 3:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        max_logit_steps = logits.size(1)
        logit_lens = torch.clamp(mel_lens // downsample, min=1)
        logit_lens = torch.minimum(
            logit_lens,
            torch.full_like(logit_lens, max_logit_steps),
        )

        logp = logits.log_softmax(-1).transpose(0, 1)
        loss = criterion(logp, targets, logit_lens, text_lens)
        losses.append(float(loss.item()))

    if model_was_training:
        model.train()

    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def diagonal_attention_score(
    model: torch.nn.Module,
    dev_loader,
    device: torch.device,
    band: float = 0.1,
    max_batches: Optional[int] = None,
) -> Tuple[float, List[float]]:
    model.eval()
    diag_scores: List[float] = []
    downsample = 2 ** getattr(model, "n_down", 1)

    for batch_idx, batch in enumerate(dev_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        texts, text_lens, mels, mel_lens = batch[:4]
        mels = mels.to(device)
        texts = texts.to(device)
        text_lens = text_lens.to(torch.long)
        mel_lens = mel_lens.to(device=device, dtype=torch.long)

        reduced_mel_lens = torch.clamp(mel_lens // downsample, min=1)
        mel_mask = model.length_to_mask(reduced_mel_lens)

        outputs = model(
            mels,
            src_key_padding_mask=mel_mask,
            text_input=texts,
        )

        attn = None
        if isinstance(outputs, dict):
            for key in ("s2s_attn", "attn", "attention", "alignments"):
                tensor = outputs.get(key)
                if tensor is not None:
                    attn = tensor
                    break
            if attn is None:
                available = ", ".join(outputs.keys())
                raise KeyError(
                    "Model output dictionary does not contain attention "
                    f"matrices. Available keys: {available}"
                )
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) < 3:
                raise ValueError(
                    "Model forward output does not include attention tensors."
                )
            attn = outputs[2]
        else:
            raise TypeError(
                "Unsupported model output type for attention extraction."
            )

        attn = attn.detach()
        time_axis = attn.size(-1)
        output_axis = attn.size(1)

        text_lens_list = text_lens.tolist()
        mel_lens_list = reduced_mel_lens.tolist()

        for b in range(attn.size(0)):
            To = min(int(text_lens_list[b]), output_axis)
            Te = min(int(mel_lens_list[b]), time_axis)
            if To <= 1 or Te <= 1:
                continue

            a = attn[b, :To, :Te]
            total_mass = a.sum()
            if torch.isclose(total_mass, torch.tensor(0.0, device=a.device)):
                continue

            t = torch.arange(To, device=a.device, dtype=torch.float32).unsqueeze(1)
            e = torch.arange(Te, device=a.device, dtype=torch.float32).unsqueeze(0)
            t = t / (To - 1) if To > 1 else torch.zeros_like(t)
            e = e / (Te - 1) if Te > 1 else torch.zeros_like(e)
            diag = t - e
            mask = (diag.abs() <= band).to(a.dtype)

            score = (a * mask).sum() / total_mass.clamp_min(1e-8)
            diag_scores.append(float(score))

    mean_score = float(np.mean(diag_scores)) if diag_scores else float("nan")
    return mean_score, diag_scores


@torch.no_grad()
def best_path_durations(logits: torch.Tensor, lens: torch.Tensor, blank_id: int):
    ids_all: List[List[int]] = []
    durs_all: List[List[int]] = []
    path = logits.argmax(-1)
    for b in range(path.size(0)):
        prev = None
        ids_b: List[int] = []
        durs_b: List[int] = []
        T = int(lens[b])
        for t in range(T):
            cur = int(path[b, t])
            if cur == blank_id:
                prev = cur
                continue
            if prev != cur:
                ids_b.append(cur)
                durs_b.append(1)
            else:
                durs_b[-1] += 1
            prev = cur
        ids_all.append(ids_b)
        durs_all.append(durs_b)
    return ids_all, durs_all


@torch.no_grad()
def duration_stats(
    model: torch.nn.Module,
    dev_loader,
    device: torch.device,
    blank_id: int,
) -> Dict[str, float]:
    model.eval()
    all_durs: List[int] = []
    ratios: List[float] = []
    downsample_factor = 2 ** getattr(model, "n_down", 1)

    for batch in dev_loader:
        texts, text_lens, mels, mel_lens = batch[:4]
        mels = mels.to(device)
        mel_lens = mel_lens.to(torch.long)

        outputs = model(mels)
        logits = _extract_ctc_logits(outputs).detach().cpu()
        logit_lens = torch.clamp(
            mel_lens // downsample_factor,
            min=1,
            max=logits.size(1),
        ).cpu()
        _, durs = best_path_durations(logits, logit_lens, blank_id)

        mel_lens_cpu = mel_lens.cpu()
        for mel_len, dur in zip(mel_lens_cpu, durs):
            if not dur:
                continue
            ratios.append(float(mel_len.item()) / len(dur))
            all_durs.extend(dur)

    if not all_durs:
        warnings.warn(
            "No non-blank token durations collected; returning NaN-filled duration stats."
        )
        return {
            "mean_dur_frames": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "frames_per_token_mean": float("nan"),
        }

    all_durs_np = np.array(all_durs, dtype=np.float32)
    ratios_np = np.array(ratios, dtype=np.float32)
    return {
        "mean_dur_frames": float(all_durs_np.mean()),
        "p50": float(np.percentile(all_durs_np, 50)),
        "p90": float(np.percentile(all_durs_np, 90)),
        "frames_per_token_mean": float(ratios_np.mean()),
    }


@torch.no_grad()
def skip_merge_flags(
    model: torch.nn.Module,
    dev_loader,
    device: torch.device,
    blank_id: int,
) -> Dict[str, float]:
    model.eval()
    diffs: List[float] = []
    downsample_factor = 2 ** getattr(model, "n_down", 1)

    for batch in dev_loader:
        texts, text_lens, mels, mel_lens = batch[:4]
        mels = mels.to(device)
        text_lens = text_lens.to(torch.long)
        mel_lens = mel_lens.to(torch.long)

        outputs = model(mels)
        logits = _extract_ctc_logits(outputs).detach().cpu()
        logit_lens = torch.clamp(
            mel_lens // downsample_factor,
            min=1,
            max=logits.size(1),
        ).cpu()

        ids, _ = best_path_durations(logits, logit_lens, blank_id)
        tgt_lens = text_lens.cpu().tolist()

        for collapsed, tlen in zip(ids, tgt_lens):
            diffs.append(len(collapsed) - int(tlen))

    if not diffs:
        raise RuntimeError(
            "No samples processed when computing skip-merge statistics."
        )

    diffs_np = np.array(diffs, dtype=np.float32)
    return {
        "mean_len_diff": float(diffs_np.mean()),
        "p10": float(np.percentile(diffs_np, 10)),
        "p90": float(np.percentile(diffs_np, 90)),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run AuxiliaryASR diagnostic probes without notebooks.",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a checkpoint file or directory containing epoch_*.pth files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a training configuration file (defaults to the one next to the checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device override (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--band",
        type=float,
        default=0.1,
        help="Diagonal attention tolerance band (default: 0.1).",
    )
    parser.add_argument(
        "--max-align-batches",
        type=int,
        default=None,
        help="Limit the number of batches when computing the diagonal score.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Number of parallel processes to use when evaluating epoch checkpoints "
            "(default: 4)."
        ),
    )
    args = parser.parse_args(argv)

    checkpoint_arg = Path(args.checkpoint)
    epoch_checkpoints: List[Tuple[int, Path]] = []
    if checkpoint_arg.is_dir():
        epoch_checkpoints = _collect_epoch_checkpoints(checkpoint_arg)

    config_path = _infer_config_path(
        checkpoint_arg, Path(args.config) if args.config else None
    )

    config = _load_config(config_path)

    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if epoch_checkpoints:
        max_workers = max(1, int(args.num_workers))
        metrics_by_epoch: Dict[int, Dict[str, float]] = {}

        executor_kwargs = {"max_workers": max_workers}
        if device.type == "cuda":
            executor_kwargs["mp_context"] = mp.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            future_to_epoch = {}
            for epoch_idx, checkpoint_file in epoch_checkpoints:
                print(
                    f"[AuxiliaryASR] dispatching evaluation for epoch {epoch_idx} "
                    f"({checkpoint_file.name})"
                )
                future_to_epoch[
                    executor.submit(
                        _evaluate_epoch_checkpoint,
                        epoch_idx,
                        str(checkpoint_file),
                        config,
                        device_str,
                        args.band,
                        args.max_align_batches,
                    )
                ] = epoch_idx

            for future in concurrent.futures.as_completed(future_to_epoch):
                epoch_idx = future_to_epoch[future]
                metrics_by_epoch[epoch_idx] = future.result()

        results: List[Tuple[int, Dict[str, float]]] = [
            (epoch_idx, metrics_by_epoch[epoch_idx])
            for epoch_idx, _ in epoch_checkpoints
        ]

        for epoch_idx, checkpoint_file in epoch_checkpoints:
            metrics = metrics_by_epoch[epoch_idx]
            print(f"=== Results for epoch {epoch_idx} ({checkpoint_file.name}) ===")
            print(
                "  PER: {per:.4f} | diagonal: {diag:.4f} | blank: {blank:.4f} | "
                "len_diff: {len_diff:.4f} | p50: {p50:.4f} | J: {joint:.4f}".format(
                    per=metrics["per"],
                    diag=metrics["diagonal"],
                    blank=metrics["blank_rate"],
                    len_diff=metrics["len_diff"],
                    p50=metrics["p50"],
                    joint=metrics["joint_score"],
                )
            )
            print()

        results.sort(key=lambda item: item[1]["joint_score"])
        top_results = results[:5]

        print("=== Top 5 checkpoints by J (lower is better) ===")
        for epoch_label, metrics in top_results:
            print(
                f"epoch {epoch_label}: J={metrics['joint_score']:.4f}, PER={metrics['per']:.4f}, "
                f"diag={metrics['diagonal']:.4f}, blank={metrics['blank_rate']:.4f}, "
                f"len_diff={metrics['len_diff']:.4f}, p50={metrics['p50']:.4f}"
            )

        return

    dev_loader, _ = build_dev_dataloader_from_config(copy.deepcopy(config), device)

    checkpoint_file = _resolve_checkpoint_path(checkpoint_arg)
    model, token_map = load_asr_model_from_config(config, str(checkpoint_file), device)

    if " " not in token_map:
        raise KeyError("The vocabulary does not contain the blank symbol ' '.")
    blank_id = int(token_map[" "])

    print("=== AuxiliaryASR_CTC_Entropy ===")
    entropy_stats = compute_ctc_entropy(model, dev_loader, device, blank_id)
    for key, value in entropy_stats.items():
        print(f"{key}: {value:.6f}")
    print()

    print("=== AuxiliaryASR_CTC_Loss ===")
    dev_loss = compute_ctc_loss(model, dev_loader, device, blank_id)
    print(f"Dev mean CTC loss: {dev_loss:.4f}")
    print()

    print("=== AuxiliaryASR_Diagonal_Attention ===")
    mean_diag, diag_scores = diagonal_attention_score(
        model,
        dev_loader,
        device,
        band=args.band,
        max_batches=args.max_align_batches,
    )
    print(f"Diagonal attention score: {mean_diag:.4f}")
    print(f"Evaluated {len(diag_scores)} alignments")
    if diag_scores:
        scores_np = np.array(diag_scores, dtype=np.float32)
        print("Score statistics:")
        print(f"  min:  {scores_np.min():.4f}")
        print(f"  25%:  {np.percentile(scores_np, 25):.4f}")
        print(f"  median: {np.median(scores_np):.4f}")
        print(f"  75%:  {np.percentile(scores_np, 75):.4f}")
        print(f"  max:  {scores_np.max():.4f}")
    print()

    print("=== AuxiliaryASR_Duration_Stats ===")
    dur_stats = duration_stats(model, dev_loader, device, blank_id)
    for key, value in dur_stats.items():
        print(f"{key}: {value:.6f}")
    print()

    print("=== AuxiliaryASR_SkipMergeFlags ===")
    skip_stats = skip_merge_flags(model, dev_loader, device, blank_id)
    for key, value in skip_stats.items():
        print(f"{key}: {value:.6f}")

    print()
    print("=== AuxiliaryASR_Phoneme_Error_Rate ===")
    per_stats = compute_phoneme_error_rate(model, dev_loader, device, blank_id)
    for key, value in per_stats.items():
        print(f"{key}: {value:.6f}")

    p50_for_penalty = dur_stats["p50"] if math.isfinite(dur_stats["p50"]) else 0.0
    joint_score = (
        per_stats["per"]
        + 0.25 * (1.0 - mean_diag)
        + 0.20 * max(0.0, entropy_stats["mean_blank_rate"] - 0.62)
        + 0.20 * max(0.0, (-skip_stats["mean_len_diff"] - 6.0) / 20.0)
        + 0.15 * max(0.0, 2.0 - p50_for_penalty)
    )
    print()
    print(f"Combined score J: {joint_score:.6f}")


if __name__ == "__main__":
    main()
