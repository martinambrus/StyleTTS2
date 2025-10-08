import math
from typing import Dict, Iterable, Optional

import torch
from torch import nn


class LegacyJDCNet(nn.Module):
    """Original JDCNet architecture bundled with StyleTTS2."""

    def __init__(self, num_class: int = 722, seq_len: int = 31, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2),
        )

        # bi-lstms
        self.bilstm_classifier = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_detector = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)
        self.detector = nn.Linear(in_features=512, out_features=2)

        self.apply(self.init_weights)

    def get_feature_GAN(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)

        return poolblock_out.transpose(-1, -2)

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)

        return self.pool_block[2](poolblock_out)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-1]
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        gan_feature = poolblock_out.transpose(-1, -2)
        poolblock_out = self.pool_block[2](poolblock_out)

        classifier_out = (
            poolblock_out.permute(0, 2, 1, 3)
            .contiguous()
            .view((-1, seq_len, 512))
        )
        classifier_out, _ = self.bilstm_classifier(classifier_out)
        classifier_out = classifier_out.contiguous().view((-1, 512))
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))

        return torch.abs(classifier_out.squeeze()), gan_feature, poolblock_out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for p in m.parameters():
                if p.data is None:
                    continue
                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with batch-first inputs."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SequenceModel(nn.Module):
    """Flexible temporal modeling block supporting BiLSTM and Transformer backends."""

    def __init__(
        self,
        input_size: int,
        model_type: str = "bilstm",
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        max_len: int = 2000,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        if self.model_type == "bilstm":
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self._output_dim = hidden_size * (2 if bidirectional else 1)
        elif self.model_type == "transformer":
            self.pos_encoding = SinusoidalPositionalEncoding(input_size, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.layer_norm = nn.LayerNorm(input_size)
            self._output_dim = input_size
        else:
            raise ValueError(f"Unsupported sequence model type: {model_type}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "bilstm":
            x, _ = self.model(x)
            return x
        if self.model_type == "transformer":
            x = self.layer_norm(self.pos_encoding(x))
            return self.model(x)
        raise RuntimeError("Invalid sequence model configuration")


class EnhancedJDCNet(nn.Module):
    """Updated JDCNet architecture from the enhanced Pitch Extractor repo."""

    def __init__(
        self,
        num_class: int = 722,
        leaky_relu_slope: float = 0.01,
        sequence_model_config: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.num_class = num_class

        sequence_model_config = dict(sequence_model_config or {})
        sequence_model_config.setdefault("model_type", "bilstm")
        sequence_model_config.setdefault("input_size", 512)
        sequence_model_config.setdefault("hidden_size", 256)
        sequence_model_config.setdefault("num_layers", 2)
        sequence_model_config.setdefault("bidirectional", True)
        sequence_model_config.setdefault("dropout", 0.0)

        self.sequence_input_dim = int(sequence_model_config.get("input_size", 512))

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )

        self.res_block1 = ResBlock(64, 128, leaky_relu_slope)
        self.res_block2 = ResBlock(128, 192, leaky_relu_slope)
        self.res_block3 = ResBlock(192, 256, leaky_relu_slope)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
        )

        classifier_cfg = dict(sequence_model_config)
        detector_cfg = dict(sequence_model_config)

        self.sequence_classifier = SequenceModel(**classifier_cfg)
        self.sequence_detector = SequenceModel(**detector_cfg)

        classifier_dim = self.sequence_classifier.output_dim
        detector_dim = self.sequence_detector.output_dim

        self.classifier = nn.Linear(classifier_dim, self.num_class)
        self.detector = nn.Linear(detector_dim, 2)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for p in m.parameters():
                if p.data is None:
                    continue
                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)

    def _prepare_sequence_input(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len = tensor.shape[2]
        return tensor.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, self.sequence_input_dim))

    def get_feature_GAN(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        poolblock_out = self.pool_block[2](poolblock_out)

        return poolblock_out.transpose(-1, -2)

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        poolblock_out = self.pool_block[2](poolblock_out)

        return self.pool_block[3](poolblock_out)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-1]
        x = x.float().transpose(-1, -2)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        poolblock_out = self.pool_block[2](poolblock_out)
        gan_feature = poolblock_out.transpose(-1, -2)
        poolblock_out = self.pool_block[3](poolblock_out)

        classifier_input = self._prepare_sequence_input(poolblock_out)
        classifier_out = self.sequence_classifier(classifier_input)
        classifier_out = classifier_out.contiguous().view((-1, classifier_out.shape[-1]))
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))

        detector_input = self._prepare_sequence_input(poolblock_out)
        detector_out = self.sequence_detector(detector_input)
        detector_out = detector_out.contiguous().view((-1, detector_out.shape[-1]))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(dim=-1)

        return torch.abs(classifier_out.squeeze(-1)), gan_feature, poolblock_out


def _collect_layer_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> Iterable[int]:
    indices = set()
    for key in state_dict:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            layer_id = suffix.split("_", 1)[0]
            if layer_id.isdigit():
                indices.add(int(layer_id))
    return sorted(indices)


def _infer_sequence_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    lstm_prefix = "sequence_classifier.model.weight_ih_l"
    transformer_prefix = "sequence_classifier.model.layers"

    if any(key.startswith(lstm_prefix) for key in state_dict):
        bidirectional = any("_reverse" in key for key in state_dict if key.startswith(lstm_prefix))
        base_key = next(key for key in state_dict if key.startswith(lstm_prefix))
        weight = state_dict[base_key]
        hidden_size = weight.shape[0] // 4
        input_size = weight.shape[1]
        layer_indices = _collect_layer_indices(state_dict, lstm_prefix)
        num_layers = max(layer_indices) + 1 if layer_indices else 1
        return {
            "model_type": "bilstm",
            "input_size": int(input_size),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "bidirectional": bidirectional,
            "dropout": 0.0,
        }

    if any(key.startswith(transformer_prefix) for key in state_dict):
        linear_key = next(
            key
            for key in state_dict
            if key.startswith("sequence_classifier.model.layers") and key.endswith("linear1.weight")
        )
        linear_weight = state_dict[linear_key]
        input_size = linear_weight.shape[1]
        dim_feedforward = linear_weight.shape[0]
        layer_ids = set()
        for key in state_dict:
            if key.startswith("sequence_classifier.model.layers"):
                parts = key.split(".")
                if len(parts) > 4 and parts[3].isdigit():
                    layer_ids.add(int(parts[3]))
        num_layers = max(layer_ids) + 1 if layer_ids else 1

        def _select_head_count(dim: int) -> int:
            for candidate in [16, 12, 10, 8, 6, 5, 4, 3, 2]:
                if candidate > 0 and dim % candidate == 0:
                    return candidate
            return 1

        nhead = _select_head_count(input_size)
        return {
            "model_type": "transformer",
            "input_size": int(input_size),
            "dim_feedforward": int(dim_feedforward),
            "num_layers": int(num_layers),
            "nhead": int(nhead),
            "dropout": 0.0,
        }

    return {
        "model_type": "bilstm",
        "input_size": 512,
        "hidden_size": 256,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.0,
    }


def build_jdc_model_from_state_dict(state_dict: Dict[str, torch.Tensor]):
    """Instantiate a pitch extractor matching the provided checkpoint."""

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None:
        num_class = classifier_weight.shape[0]
    else:
        num_class = 1

    if any(key.startswith("sequence_classifier") for key in state_dict):
        sequence_config = _infer_sequence_model_config(state_dict)
        return EnhancedJDCNet(num_class=num_class, sequence_model_config=sequence_config)

    return LegacyJDCNet(num_class=num_class)


__all__ = [
    "LegacyJDCNet",
    "EnhancedJDCNet",
    "build_jdc_model_from_state_dict",
]
