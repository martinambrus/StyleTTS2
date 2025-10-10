import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from .layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock


def _stochastic_depth(residual: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Apply sample-wise stochastic depth to a residual branch."""

    if drop_prob <= 0.0 or not training:
        return residual

    keep_prob = 1.0 - drop_prob
    if keep_prob <= 0.0:
        return torch.zeros_like(residual)

    shape = (residual.size(0),) + (1,) * (residual.dim() - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=residual.dtype, device=residual.device)
    random_tensor.floor_()
    return residual / keep_prob * random_tensor


class EncoderStage(nn.Module):
    """A convolutional encoder stage with optional stochastic depth."""

    def __init__(self, hidden_dim: int, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(max(0.0, drop_prob))
        self.block = ConvBlock(hidden_dim)
        self.post_norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.post_norm(residual)

        if self.drop_prob > 0.0:
            delta = residual - x
            delta = _stochastic_depth(delta, self.drop_prob, self.training)
            residual = x + delta

        return residual


class IntermediateCTCHead(nn.Module):
    """Light-weight projection head for intermediate CTC supervision."""

    def __init__(self, hidden_dim: int, n_token: int, dropout: float = 0.1):
        super().__init__()
        projection_dim = max(1, hidden_dim // 2)
        self.layers = nn.Sequential(
            ConvNorm(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ConvNorm(projection_dim, n_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)
        return logits.transpose(1, 2)


class SelfConditionedCTCBlock(nn.Module):
    """Predicts self-conditioned CTC distributions and feeds them back to the encoder."""

    def __init__(
        self,
        hidden_dim: int,
        n_token: int,
        strategy: str = "add",
        detach_conditioning: bool = True,
        temperature: float = 1.0,
        predictor_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.strategy = str(strategy).lower()
        if self.strategy not in {"add", "concat"}:
            raise ValueError(f"Unsupported self-conditioned strategy: {self.strategy}")

        self.detach_conditioning = bool(detach_conditioning)
        self.temperature = max(1e-5, float(temperature))

        projection_dim = max(1, hidden_dim // 2)
        self.predictor = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(predictor_dropout),
            ConvNorm(hidden_dim, projection_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(predictor_dropout),
            ConvNorm(projection_dim, n_token, kernel_size=1),
        )

        self.condition_projector = nn.Sequential(
            nn.Dropout(predictor_dropout),
            ConvNorm(n_token, hidden_dim, kernel_size=1),
        )

        if self.strategy == "concat":
            self.fusion = nn.Sequential(
                nn.Dropout(fusion_dropout),
                ConvNorm(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.GELU(),
            )
        else:
            self.fusion = None

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return conditioned features along with logits and log-probabilities."""

        logits = self.predictor(features)
        scaled_logits = logits / self.temperature
        log_probs = F.log_softmax(scaled_logits, dim=1)
        probs = log_probs.exp()

        conditioning_source = probs.detach() if self.detach_conditioning else probs
        conditioning = self.condition_projector(conditioning_source)

        if self.strategy == "concat" and self.fusion is not None:
            fused = torch.cat([features, conditioning], dim=1)
            conditioned_features = self.fusion(fused)
        else:
            conditioned_features = features + conditioning

        return {
            "features": conditioned_features,
            "logits": logits.transpose(1, 2),
            "log_probs": log_probs.transpose(1, 2),
        }


def build_model(model_params={}, model_type='asr'):
    model = ASRCNN(**model_params)
    return model


class ASRCNN(nn.Module):
    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        n_token=35,
        n_layers=6,
        token_embedding_dim=256,
        location_kernel_size=63,
        attention_dropout=0.0,
        multi_task_config=None,
        stabilization_config=None,
        memory_optimization_config=None,
    ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(input_dim // 2, hidden_dim, kernel_size=7, padding=3, stride=2)
        self.stabilization_config = stabilization_config or {}
        self._stochastic_depth_cfg = self.stabilization_config.get('stochastic_depth', {}) or {}
        self.enable_stochastic_depth = bool(self._stochastic_depth_cfg.get('enabled', False))
        self.memory_optimization_config = memory_optimization_config or {}
        self._gradient_checkpoint_cfg = self.memory_optimization_config.get('gradient_checkpointing', {}) or {}
        self.enable_gradient_checkpointing = bool(self._gradient_checkpoint_cfg.get('enabled', False))

        def _safe_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        self.gradient_checkpoint_start_layer = max(1, _safe_int(self._gradient_checkpoint_cfg.get('start_layer', 1), 1))
        end_layer = self._gradient_checkpoint_cfg.get('end_layer', None)
        parsed_end = None if end_layer in (None, '', 'none', 'null') else _safe_int(end_layer, None)
        if isinstance(parsed_end, int) and parsed_end <= 0:
            parsed_end = None
        if isinstance(parsed_end, int):
            parsed_end = max(self.gradient_checkpoint_start_layer, min(n_layers, parsed_end))
        self.gradient_checkpoint_end_layer: Optional[int] = parsed_end
        segments = _safe_int(self._gradient_checkpoint_cfg.get('segments', 1), 1)
        self.gradient_checkpoint_segments = max(1, segments)
        chunk_size = _safe_int(self._gradient_checkpoint_cfg.get('chunk_size', 0), 0)
        self.gradient_checkpoint_chunk_size = max(0, chunk_size)
        min_seq_len = _safe_int(self._gradient_checkpoint_cfg.get('min_sequence_length', 0), 0)
        self.gradient_checkpoint_min_seq_len = max(0, min_seq_len)
        self.gradient_checkpoint_use_sequential = bool(
            self._gradient_checkpoint_cfg.get('use_checkpoint_sequential', True)
        )
        self.gradient_checkpoint_use_reentrant = bool(
            self._gradient_checkpoint_cfg.get('use_reentrant', False)
        )

        self.encoder_layers = nn.ModuleList()
        for layer_idx in range(1, n_layers + 1):
            if self.enable_stochastic_depth:
                drop_prob = self._get_stochastic_depth_prob(layer_idx, n_layers)
            else:
                drop_prob = 0.0
            self.encoder_layers.append(EncoderStage(hidden_dim, drop_prob=drop_prob))

        ictc_cfg = self.stabilization_config.get('intermediate_ctc', {}) or {}
        self.enable_intermediate_ctc = bool(ictc_cfg.get('enabled', False))
        ictc_dropout = float(ictc_cfg.get('dropout', 0.1))
        self.intermediate_ctc_layers = self._parse_intermediate_layers(ictc_cfg.get('layers'), n_layers)
        if self.enable_intermediate_ctc and self.intermediate_ctc_layers:
            self.intermediate_ctc_heads = nn.ModuleDict({
                str(layer_idx): IntermediateCTCHead(hidden_dim, n_token, dropout=ictc_dropout)
                for layer_idx in self.intermediate_ctc_layers
            })
        else:
            self.intermediate_ctc_heads = nn.ModuleDict()

        sctc_cfg = self.stabilization_config.get('self_conditioned_ctc', {}) or {}
        self.enable_self_conditioned_ctc = bool(sctc_cfg.get('enabled', False))
        self.self_conditioning_layers = self._parse_intermediate_layers(sctc_cfg.get('layers'), n_layers)
        if self.enable_self_conditioned_ctc and self.self_conditioning_layers:
            self.self_conditioning_blocks = nn.ModuleDict({
                str(layer_idx): SelfConditionedCTCBlock(
                    hidden_dim=hidden_dim,
                    n_token=n_token,
                    strategy=sctc_cfg.get('conditioning_strategy', 'add'),
                    detach_conditioning=sctc_cfg.get('detach_conditioning', True),
                    temperature=sctc_cfg.get('temperature', 1.0),
                    predictor_dropout=sctc_cfg.get('predictor_dropout', 0.1),
                    fusion_dropout=sctc_cfg.get('fusion_dropout', 0.1),
                )
                for layer_idx in self.self_conditioning_layers
            })
        else:
            self.self_conditioning_blocks = nn.ModuleDict()

        self._checkpoint_special_layers = set(self.intermediate_ctc_layers)
        self._checkpoint_special_layers.update(self.self_conditioning_layers)

        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.multi_task_config = multi_task_config or {}
        self.use_ctc = bool(self.multi_task_config.get('use_ctc', True))
        self.use_seq2seq = bool(self.multi_task_config.get('use_seq2seq', True))

        head_sharing_cfg = (self.multi_task_config.get('head_sharing', {}) or {})
        ctc_seq2seq_cfg = (head_sharing_cfg.get('ctc_seq2seq', {}) or {})
        self.enable_ctc_seq2seq_sharing = bool(
            ctc_seq2seq_cfg.get('enabled', False)
            and self.use_ctc
            and self.use_seq2seq
        )
        self.ctc_seq2seq_detach = bool(ctc_seq2seq_cfg.get('detach_for_seq2seq', False))

        self.ctc_state_projector: Optional[nn.Module] = None
        self.ctc_state_activation: Optional[nn.Module] = None
        self.ctc_classifier: Optional[nn.Module] = None
        self.ctc_seq2seq_adapter: Optional[nn.Module] = None

        if self.use_ctc:
            if self.enable_ctc_seq2seq_sharing:
                self.ctc_state_projector = LinearNorm(hidden_dim // 2, hidden_dim)
                self.ctc_state_activation = nn.ReLU()
                self.ctc_classifier = LinearNorm(hidden_dim, n_token)
                self.ctc_seq2seq_adapter = LinearNorm(hidden_dim, hidden_dim // 2)
                self.ctc_linear = None
            else:
                self.ctc_linear = nn.Sequential(
                    LinearNorm(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    LinearNorm(hidden_dim, n_token))
        else:
            self.ctc_linear = None

        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token,
            location_kernel_size=location_kernel_size,
            attention_dropout=attention_dropout)

        duration_hidden = max(4, hidden_dim // 16)
        self.duration_predictor = nn.Sequential(
            nn.Embedding(n_token, duration_hidden),
            nn.ReLU(),
            nn.Linear(duration_hidden, 1),
            nn.Softplus(),
        )

        frame_cfg = self.multi_task_config.get('frame_phoneme', {}) or {}
        self.enable_frame_classifier = bool(frame_cfg.get('enabled', False))
        if self.enable_frame_classifier:
            n_classes = int(frame_cfg.get('num_classes') or 0)
            if n_classes <= 0:
                n_classes = n_token
            self.frame_classifier = nn.Sequential(
                LinearNorm(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                LinearNorm(hidden_dim // 2, n_classes)
            )
            self.frame_num_classes = n_classes
        else:
            self.frame_classifier = None
            self.frame_num_classes = 0

        speaker_cfg = self.multi_task_config.get('speaker', {}) or {}
        self.enable_speaker = bool(speaker_cfg.get('enabled', False))
        if self.enable_speaker:
            embedding_dim = int(speaker_cfg.get('embedding_dim', hidden_dim // 2))
            self.num_speakers = max(1, int(speaker_cfg.get('num_speakers', 1)))
            self.speaker_projection = nn.Linear(hidden_dim // 2, embedding_dim)
            self.speaker_norm = nn.LayerNorm(embedding_dim)
            self.speaker_classifier = nn.Linear(embedding_dim, self.num_speakers)
        else:
            self.speaker_projection = None
            self.speaker_classifier = None
            self.speaker_norm = None
            self.num_speakers = 0

        pron_cfg = self.multi_task_config.get('pronunciation_error', {}) or {}
        self.enable_pronunciation_error = bool(pron_cfg.get('enabled', False))
        if self.enable_pronunciation_error:
            num_classes = max(2, int(pron_cfg.get('num_classes', 2)))
            self.pron_error_head = nn.Sequential(
                LinearNorm(self.asr_s2s.decoder_rnn_dim, hidden_dim // 2),
                nn.ReLU(),
                LinearNorm(hidden_dim // 2, num_classes)
            )
            self.pron_error_num_classes = num_classes
        else:
            self.pron_error_head = None
            self.pron_error_num_classes = 0

    def _get_stochastic_depth_prob(self, layer_idx: int, total_layers: int) -> float:
        strategy = str(self._stochastic_depth_cfg.get('mode', 'linear')).lower()
        min_drop = float(self._stochastic_depth_cfg.get('min_drop_rate', 0.0))
        max_drop = float(self._stochastic_depth_cfg.get('max_drop_rate', self._stochastic_depth_cfg.get('drop_rate', 0.0)))
        max_drop = max(0.0, min(1.0, max_drop))
        min_drop = max(0.0, min(1.0, min_drop))
        if total_layers <= 1:
            return max_drop

        if strategy == 'uniform':
            return max_drop

        progress = (layer_idx - 1) / (total_layers - 1)
        drop = min_drop + (max_drop - min_drop) * progress
        return max(0.0, min(1.0, drop))

    @staticmethod
    def _parse_intermediate_layers(layers_config, max_layers: int) -> List[int]:
        if layers_config is None:
            return []

        parsed: List[int] = []
        if isinstance(layers_config, dict):
            source = layers_config.keys()
        else:
            source = layers_config

        for entry in source:
            if isinstance(entry, dict):
                idx = entry.get('index', entry.get('layer'))
            else:
                idx = entry
            try:
                value = int(idx)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= max_layers:
                parsed.append(value)

        seen = set()
        ordered: List[int] = []
        for value in parsed:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        intermediate_outputs: Dict[str, torch.Tensor] = {}
        self_conditioned_outputs: Dict[str, torch.Tensor] = {}
        self_conditioned_log_probs: Dict[str, torch.Tensor] = {}

        chunk: List[Tuple[int, nn.Module]] = []
        chunk_input: Optional[torch.Tensor] = None

        def _flush_chunk() -> Optional[int]:
            nonlocal x, chunk, chunk_input
            if not chunk:
                return None

            modules = [module for _, module in chunk]
            segments = min(len(modules), self.gradient_checkpoint_segments)
            segments = max(1, segments)

            input_tensor = chunk_input if chunk_input is not None else x

            if self.gradient_checkpoint_use_sequential or len(modules) > 1:
                try:
                    x = checkpoint_sequential(
                        modules,
                        segments,
                        input_tensor,
                        use_reentrant=self.gradient_checkpoint_use_reentrant,
                    )
                except TypeError:
                    x = checkpoint_sequential(modules, segments, input_tensor)
            else:
                x = checkpoint(
                    modules[0], input_tensor, use_reentrant=self.gradient_checkpoint_use_reentrant
                )

            _, last_module = chunk[-1]
            chunk = []
            chunk_input = None
            return id(last_module)

        active_checkpoint_id = None

        for layer_idx, layer in enumerate(self.encoder_layers, start=1):
            within_range = False
            if self.enable_gradient_checkpointing:
                within_range = layer_idx >= self.gradient_checkpoint_start_layer and (
                    self.gradient_checkpoint_end_layer is None
                    or layer_idx <= self.gradient_checkpoint_end_layer
                )
                if within_range:
                    if not chunk:
                        chunk_input = x
                    chunk.append((layer_idx, layer))
                    active_checkpoint_id = id(layer)
                elif chunk:
                    active_checkpoint_id = _flush_chunk()

            output = layer(x)
            if self.enable_intermediate_ctc and str(layer_idx) in self.intermediate_ctc_heads:
                head = self.intermediate_ctc_heads[str(layer_idx)]
                intermediate_outputs[str(layer_idx)] = head(output)
            if self.enable_self_conditioned_ctc and str(layer_idx) in self.self_conditioning_blocks:
                block = self.self_conditioning_blocks[str(layer_idx)]
                conditioned = block(output)
                output = conditioned['features']
                self_conditioned_outputs[str(layer_idx)] = conditioned['logits']
                self_conditioned_log_probs[str(layer_idx)] = conditioned['log_probs']

            if within_range:
                x = output
                continue

            if chunk:
                active_checkpoint_id = _flush_chunk()

            x = output

        if chunk:
            active_checkpoint_id = _flush_chunk()

        x = self.projection(x)
        x = x.transpose(1, 2)

        ctc_logit = None
        ctc_hidden = None
        if self.use_ctc:
            if self.enable_ctc_seq2seq_sharing and self.ctc_state_projector is not None:
                ctc_hidden = self.ctc_state_projector(x)
                if self.ctc_state_activation is not None:
                    ctc_hidden = self.ctc_state_activation(ctc_hidden)
                if self.ctc_classifier is not None:
                    ctc_logit = self.ctc_classifier(ctc_hidden)
                else:
                    ctc_logit = None
            elif self.ctc_linear is not None:
                ctc_logit = self.ctc_linear(x)

        if ctc_hidden is None:
            ctc_hidden = x

        seq2seq_input = ctc_hidden
        if self.enable_ctc_seq2seq_sharing and self.ctc_seq2seq_adapter is not None:
            adapted = self.ctc_seq2seq_adapter(ctc_hidden)
            if self.ctc_seq2seq_detach:
                adapted = adapted.detach()
            seq2seq_input = adapted

        if text_input is not None and self.use_seq2seq:
            _, s2s_logit, s2s_attn = self.asr_s2s(seq2seq_input, src_key_padding_mask, text_input)
        else:
            s2s_logit = None
            s2s_attn = None

        outputs = {
            'ctc': ctc_logit,
            'seq2seq': s2s_logit,
            'attention': s2s_attn,
            'duration': self.duration_predictor(text_input) if text_input is not None else None,
            'intermediate_ctc': intermediate_outputs,
            'self_conditioned_ctc': self_conditioned_outputs,
            'self_conditioned_log_probs': self_conditioned_log_probs,
        }

        if self.enable_frame_classifier:
            outputs['frame_phoneme'] = self.frame_classifier(x)
        else:
            outputs['frame_phoneme'] = None

        if self.enable_speaker:
            speaker_embed = self.speaker_projection(x)
            if self.speaker_norm is not None:
                speaker_embed = self.speaker_norm(speaker_embed)
            outputs['speaker_logits'] = self.speaker_classifier(speaker_embed.mean(dim=1))
        else:
            outputs['speaker_logits'] = None

        if self.enable_pronunciation_error and s2s_logit is not None:
            outputs['pronunciation_error'] = self.pron_error_head(s2s_logit)
        else:
            outputs['pronunciation_error'] = None

        self.last_outputs = outputs
        return outputs['ctc'], outputs['seq2seq'], outputs['attention']

    def get_feature(self, x):
        x = self.to_mfcc(x.squeeze(1))
        x = self.init_cnn(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1)).to(lengths.device)
        return mask

    def get_future_mask(self, out_length, unmask_future_steps=0):
        index_tensor = torch.arange(out_length).unsqueeze(0).expand(out_length, -1)
        mask = torch.gt(index_tensor, index_tensor.T + unmask_future_steps)
        return mask


class ASRS2S(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        hidden_dim=512,
        n_location_filters=32,
        location_kernel_size=63,
        n_token=40,
        attention_dropout=0.0,
    ):
        super(ASRS2S, self).__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size,
            attention_dropout=attention_dropout,
        )
        self.decoder_rnn = nn.LSTMCell(self.decoder_rnn_dim + embedding_dim, self.decoder_rnn_dim)
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim),
            nn.Tanh())
        self.sos = 1
        self.eos = 2

    def initialize_decoder_states(self, memory, mask):
        B, L, H = memory.shape
        self.decoder_hidden = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.decoder_cell = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.attention_weights = torch.zeros((B, L)).type_as(memory)
        self.attention_weights_cum = torch.zeros((B, L)).type_as(memory)
        self.attention_context = torch.zeros((B, H)).type_as(memory)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        self.unk_index = 3
        self.random_mask = 0.1

    def forward(self, memory, memory_mask, text_input):
        self.initialize_decoder_states(memory, memory_mask)
        random_mask = (torch.rand(text_input.shape) < self.random_mask).to(text_input.device)
        _text_input = text_input.clone()
        _text_input.masked_fill_(random_mask, self.unk_index)
        decoder_inputs = self.embedding(_text_input).transpose(0, 1)
        start_embedding = self.embedding(
            torch.LongTensor([self.sos] * decoder_inputs.size(1)).to(decoder_inputs.device))
        decoder_inputs = torch.cat((start_embedding.unsqueeze(0), decoder_inputs), dim=0)

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):
            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]

        hidden_outputs, logit_outputs, alignments = \
            self.parse_decoder_outputs(
                hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input,
            (self.decoder_hidden, self.decoder_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask)

        self.attention_weights_cum += self.attention_weights

        hidden_and_context = torch.cat((self.decoder_hidden, self.attention_context), -1)
        hidden = self.project_to_hidden(hidden_and_context)

        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self, hidden, logit, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()

        return hidden, logit, alignments
