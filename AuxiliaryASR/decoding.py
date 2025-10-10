"""Decoding utilities for AuxiliaryASR.

This module implements a configurable CTC beam search decoder with optional
language model integration.  Two LM fusion strategies are provided:

* **Shallow fusion** with an n-gram language model operating on phoneme tokens.
* **Cold fusion** with a neural language model whose hidden state is consulted
  through a lightweight gating mechanism.

Both strategies can be enabled or disabled independently via configuration and
share a common interface so the decoder can keep track of the LM state for each
active hypothesis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _log_sum_exp(a: float, b: float) -> float:
    """Stable log-sum-exp for two scalar log probabilities."""

    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class Hypothesis:
    """Represents a single beam search hypothesis."""

    prefix: Tuple[int, ...]
    log_prob_blank: float = -math.inf
    log_prob_non_blank: float = -math.inf
    shallow_state: Optional[Tuple[int, ...]] = None
    neural_state: Optional["NeuralLMState"] = None

    @property
    def score(self) -> float:
        return _log_sum_exp(self.log_prob_blank, self.log_prob_non_blank)


class PhonemeNGramLM:
    """Simple phoneme level n-gram language model for shallow fusion.

    The model reads counts from a whitespace separated text file where each
    line represents an n-gram followed by its count, e.g. ``1 3 5\t42``.  The
    final element is interpreted as a frequency count and converted to log
    probabilities with add-k smoothing.
    """

    def __init__(
        self,
        path: Optional[Path],
        order: int,
        vocab_size: int,
        smoothing: float = 1.0,
    ) -> None:
        self.order = max(1, int(order))
        self.vocab_size = int(vocab_size)
        self.smoothing = float(smoothing)
        self.ngram_counts: Dict[Tuple[int, ...], float] = defaultdict(float)
        self.context_totals: Dict[Tuple[int, ...], float] = defaultdict(
            lambda: self.smoothing * self.vocab_size
        )

        if path is not None:
            self._load_counts(path)

    def _load_counts(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Phoneme n-gram LM file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    ngram, count_str = line.split("\t", 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    ngram, count_str = " ".join(parts[:-1]), parts[-1]

                tokens = tuple(int(tok) for tok in ngram.split())
                if not tokens:
                    continue
                count = float(count_str)
                tokens = tokens[-self.order :]
                context = tokens[:-1]
                self.ngram_counts[tokens] += count
                self.context_totals[context] += count

    def start(self) -> Tuple[int, ...]:
        return tuple()

    def log_prob(self, state: Tuple[int, ...], token: int) -> Tuple[float, Tuple[int, ...]]:
        context = tuple(state[-(self.order - 1) :]) if self.order > 1 else tuple()
        ngram = context + (token,)
        count = self.ngram_counts.get(ngram, 0.0) + self.smoothing
        total = self.context_totals.get(context, self.smoothing * self.vocab_size)
        prob = math.log(max(count, 1e-12)) - math.log(max(total, 1e-12))
        new_state = ngram[-(self.order - 1) :] if self.order > 1 else tuple()
        return prob, new_state


@dataclass
class NeuralLMState:
    hidden: torch.Tensor
    log_probs: torch.Tensor


class NeuralLanguageModel(nn.Module):
    """Minimal neural LM used for cold fusion.

    The network is deliberately lightweight (Embedding -> GRU -> Linear) and
    exposes utilities to keep its hidden state on the Python side so the beam
    search can request incremental token probabilities.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        sos_id: int = 1,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.sos_id = int(sos_id)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward_step(
        self, token: torch.Tensor, hidden: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token.unsqueeze(0))  # [1, B, E]
        out, new_hidden = self.rnn(emb, hidden)
        logits = self.output(out.squeeze(0))
        log_probs = F.log_softmax(logits, dim=-1)
        return new_hidden, log_probs

    def start(self, batch_size: int = 1) -> NeuralLMState:
        hidden = torch.zeros(
            self.rnn.num_layers,
            batch_size,
            self.rnn.hidden_size,
            device=self.device,
        )
        token = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=self.device)
        new_hidden, log_probs = self.forward_step(token, hidden)
        return NeuralLMState(hidden=new_hidden, log_probs=log_probs)

    def score(
        self, state: NeuralLMState, token: int
    ) -> Tuple[float, NeuralLMState]:
        log_prob = state.log_probs[..., token].item()
        token_tensor = torch.tensor([token], dtype=torch.long, device=self.device)
        new_hidden, next_log_probs = self.forward_step(token_tensor, state.hidden)
        return log_prob, NeuralLMState(hidden=new_hidden, log_probs=next_log_probs)


class ColdFusionCombiner:
    """Implements a lightweight cold fusion gating mechanism."""

    def __init__(
        self,
        fusion_weight: float = 0.5,
        gate_bias: float = 0.0,
        gate_scale: float = 1.0,
    ) -> None:
        self.fusion_weight = float(fusion_weight)
        self.gate_bias = float(gate_bias)
        self.gate_scale = float(gate_scale)

    def gate(self, state: NeuralLMState) -> float:
        if state.hidden is None:
            return 1.0 / (1.0 + math.exp(-self.gate_bias))
        hidden_mean = state.hidden.abs().mean().item()
        activation = self.gate_bias + self.gate_scale * hidden_mean
        return 1.0 / (1.0 + math.exp(-activation))

    def fusion_bonus(self, state: NeuralLMState, lm_log_prob: float) -> float:
        gate = self.gate(state)
        return gate * self.fusion_weight * lm_log_prob


class CTCBeamSearchDecoder:
    """Performs CTC beam search with optional LM integration."""

    def __init__(
        self,
        beam_width: int,
        blank_id: int,
        log_probs_input: bool = False,
        shallow_fusion_lm: Optional[PhonemeNGramLM] = None,
        shallow_weight: float = 0.0,
        cold_fusion_lm: Optional[NeuralLanguageModel] = None,
        cold_fusion: Optional[ColdFusionCombiner] = None,
        length_penalty: float = 0.0,
        prune_threshold: float = 1e-6,
        logit_temperature: float = 1.0,
        blank_penalty: float = 0.0,
        insertion_bonus: float = 0.0,
    ) -> None:
        self.beam_width = max(1, int(beam_width))
        self.blank_id = int(blank_id)
        self.log_probs_input = bool(log_probs_input)
        self.shallow_fusion_lm = shallow_fusion_lm
        self.shallow_weight = float(shallow_weight)
        self.cold_fusion_lm = cold_fusion_lm
        self.cold_fusion = cold_fusion
        self.length_penalty = float(length_penalty)
        self.prune_threshold = float(prune_threshold)
        self.logit_temperature = max(1.0e-6, float(logit_temperature))
        self.blank_penalty = float(blank_penalty)
        self.insertion_bonus = float(insertion_bonus)

    def _init_hypothesis(self) -> Hypothesis:
        shallow_state = self.shallow_fusion_lm.start() if self.shallow_fusion_lm else None
        neural_state = self.cold_fusion_lm.start() if self.cold_fusion_lm else None
        return Hypothesis(
            prefix=tuple(),
            log_prob_blank=0.0,
            log_prob_non_blank=-math.inf,
            shallow_state=shallow_state,
            neural_state=neural_state,
        )

    def _prune(self, probs: torch.Tensor) -> Iterable[int]:
        if self.prune_threshold <= 0.0:
            return range(probs.size(0))
        threshold = probs.max().item() + math.log(self.prune_threshold)
        return [i for i, log_p in enumerate(probs.tolist()) if log_p >= threshold]

    def _length_normalize(self, score: float, length: int) -> float:
        if self.length_penalty == 0 or length == 0:
            return score
        return score / (length ** self.length_penalty)

    def decode_single(self, logits: torch.Tensor, seq_len: int) -> List[int]:
        if not self.log_probs_input:
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            log_probs = logits

        if self.logit_temperature != 1.0:
            log_probs = log_probs / self.logit_temperature
            log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

        beams: Dict[Tuple[int, ...], Hypothesis] = {
            tuple(): self._init_hypothesis()
        }

        for t in range(seq_len):
            frame = log_probs[t]
            candidates: Dict[Tuple[int, ...], Hypothesis] = {}
            active_symbols = self._prune(frame)

            sorted_beams = sorted(beams.values(), key=lambda h: h.score, reverse=True)[
                : self.beam_width
            ]

            for hyp in sorted_beams:
                total_log_prob = hyp.score

                # Extend with blank
                blank_log_prob = frame[self.blank_id].item() - self.blank_penalty
                new_blank_prob = total_log_prob + blank_log_prob
                updated = candidates.setdefault(hyp.prefix, Hypothesis(prefix=hyp.prefix))
                updated.log_prob_blank = _log_sum_exp(updated.log_prob_blank, new_blank_prob)
                if updated.shallow_state is None:
                    updated.shallow_state = hyp.shallow_state
                if updated.neural_state is None:
                    updated.neural_state = hyp.neural_state

                for symbol in active_symbols:
                    if symbol == self.blank_id:
                        continue

                    symbol_log_prob = frame[symbol].item() + self.insertion_bonus
                    prev_symbol = hyp.prefix[-1] if hyp.prefix else None

                    shallow_state = hyp.shallow_state
                    shallow_bonus = 0.0
                    if self.shallow_fusion_lm is not None:
                        shallow_bonus, new_shallow_state = self.shallow_fusion_lm.log_prob(
                            hyp.shallow_state or tuple(), symbol
                        )
                    else:
                        new_shallow_state = shallow_state

                    neural_state = hyp.neural_state
                    cold_bonus = 0.0
                    if self.cold_fusion_lm is not None and self.cold_fusion is not None:
                        if neural_state is None:
                            neural_state = self.cold_fusion_lm.start()

                        if 0 <= symbol < self.cold_fusion_lm.vocab_size:
                            lm_log_prob, new_neural_state = self.cold_fusion_lm.score(
                                neural_state, symbol
                            )
                            cold_bonus = self.cold_fusion.fusion_bonus(
                                new_neural_state, lm_log_prob
                            )
                        else:
                            lm_log_prob = None
                            new_neural_state = neural_state
                    else:
                        lm_log_prob = None
                        new_neural_state = neural_state

                    next_prefix: Tuple[int, ...]
                    prob_non_blank: float
                    if symbol == prev_symbol:
                        prob_non_blank = hyp.log_prob_non_blank + symbol_log_prob
                        next_prefix = hyp.prefix
                    else:
                        total = _log_sum_exp(hyp.log_prob_blank, hyp.log_prob_non_blank)
                        prob_non_blank = total + symbol_log_prob
                        next_prefix = hyp.prefix + (symbol,)

                    prob_non_blank += self.shallow_weight * shallow_bonus
                    if cold_bonus != 0.0:
                        prob_non_blank += cold_bonus

                    candidate = candidates.setdefault(
                        next_prefix,
                        Hypothesis(prefix=next_prefix),
                    )

                    candidate.log_prob_non_blank = _log_sum_exp(
                        candidate.log_prob_non_blank, prob_non_blank
                    )
                    candidate.shallow_state = new_shallow_state
                    candidate.neural_state = new_neural_state

            beams = candidates

        best = max(beams.values(), key=lambda h: self._length_normalize(h.score, len(h.prefix)))
        return list(best.prefix)

    def decode(self, logits: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        if logits.dim() != 3:
            raise ValueError(
                f"Expected logits of shape (B, T, V), got {tuple(logits.shape)}"
            )

        results: List[List[int]] = []
        for b in range(logits.size(0)):
            seq_len = int(lengths[b])
            results.append(self.decode_single(logits[b], seq_len))
        return results


def build_decoder_from_config(config: dict) -> Optional[CTCBeamSearchDecoder]:
    decoding_cfg = config.get("decoding", {}) or {}
    beam_cfg = decoding_cfg.get("beam_search", {}) or {}
    if not beam_cfg.get("enabled", False):
        return None

    shallow_cfg = decoding_cfg.get("shallow_fusion", {}) or {}
    shallow_lm = None
    shallow_weight = float(shallow_cfg.get("lm_weight", 0.0))
    if shallow_cfg.get("enabled", False):
        lm_path = shallow_cfg.get("lm_path")
        order = shallow_cfg.get("order", 3)
        vocab_size = shallow_cfg.get("vocab_size")
        if vocab_size is None:
            raise ValueError("shallow_fusion.vocab_size must be specified when enabled")
        shallow_lm = PhonemeNGramLM(
            Path(lm_path) if lm_path else None,
            order=order,
            vocab_size=vocab_size,
            smoothing=shallow_cfg.get("smoothing", 1.0),
        )

    cold_cfg = decoding_cfg.get("cold_fusion", {}) or {}
    cold_lm = None
    cold_combiner = None
    if cold_cfg.get("enabled", False):
        neural_cfg = cold_cfg.get("neural_lm", {}) or {}
        vocab_size = neural_cfg.get("vocab_size")
        if vocab_size is None:
            raise ValueError("cold_fusion.neural_lm.vocab_size must be set when enabled")
        cold_lm = NeuralLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=neural_cfg.get("embedding_dim", 128),
            hidden_dim=neural_cfg.get("hidden_dim", 256),
            num_layers=neural_cfg.get("num_layers", 1),
            sos_id=neural_cfg.get("sos_id", 1),
            dropout=neural_cfg.get("dropout", 0.0),
        )
        checkpoint = neural_cfg.get("checkpoint")
        if checkpoint:
            state = torch.load(checkpoint, map_location=cold_lm.device, weights_only=False)
            state_dict = state.get("state_dict", state)
            cold_lm.load_state_dict(state_dict)
        cold_lm.eval()
        cold_combiner = ColdFusionCombiner(
            fusion_weight=cold_cfg.get("fusion_weight", 0.5),
            gate_bias=cold_cfg.get("gate_bias", 0.0),
            gate_scale=cold_cfg.get("gate_scale", 1.0),
        )

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_cfg.get("beam_width", 10),
        blank_id=beam_cfg.get("blank_id", 0),
        log_probs_input=beam_cfg.get("log_probs_input", False),
        shallow_fusion_lm=shallow_lm,
        shallow_weight=shallow_weight,
        cold_fusion_lm=cold_lm,
        cold_fusion=cold_combiner,
        length_penalty=beam_cfg.get("length_penalty", 0.0),
        prune_threshold=beam_cfg.get("prune_threshold", 1e-6),
        logit_temperature=beam_cfg.get("logit_temperature", 1.0),
        blank_penalty=beam_cfg.get("blank_penalty", 0.0),
        insertion_bonus=beam_cfg.get("insertion_bonus", 0.0),
    )
    return decoder


__all__ = [
    "CTCBeamSearchDecoder",
    "PhonemeNGramLM",
    "NeuralLanguageModel",
    "ColdFusionCombiner",
    "build_decoder_from_config",
]

