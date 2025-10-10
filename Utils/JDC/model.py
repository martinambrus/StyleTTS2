import math
from typing import Dict, Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with batch-first inputs."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


class LegacyJDCNet(nn.Module):
    """Original JDCNet architecture used by legacy checkpoints."""

    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )

        self.res_block1 = ResBlock(in_channels=64, out_channels=128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.2),
        )

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

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)

    def forward(self, x):
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
            poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        classifier_out, _ = self.bilstm_classifier(classifier_out)
        classifier_out = classifier_out.contiguous().view((-1, 512))
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))

        mp1_out = self.maxpool1(convblock_out)
        mp2_out = self.maxpool2(resblock1_out)
        mp3_out = self.maxpool3(resblock2_out)

        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        detector_out = (
            detector_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        detector_out, _ = self.bilstm_detector(detector_out)
        detector_out = detector_out.contiguous().view((-1, 512))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)

        f0 = classifier_out
        if f0.dim() >= 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
        f0 = torch.abs(f0)
        return f0, detector_out, poolblock_out


class ModernJDCNet(nn.Module):
    """Updated JDCNet architecture matching the modern training code."""

    def __init__(
        self,
        num_class=1,
        leaky_relu_slope=0.01,
        sequence_model_config: Optional[Dict] = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_class = num_class
        sequence_model_config = sequence_model_config or {}
        head_dropout = max(0.1, min(0.3, float(head_dropout)))
        self.classifier_dropout = nn.Dropout(p=head_dropout)
        self.detector_dropout = nn.Dropout(p=head_dropout)

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )

        self.res_block1 = ResBlock(in_channels=64, out_channels=128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )

        sequence_model_config = dict(sequence_model_config)
        sequence_model_config.setdefault("input_size", 512)
        classifier_config = dict(sequence_model_config)
        detector_config = dict(sequence_model_config)
        self.sequence_classifier = SequenceModel(**classifier_config)
        self.sequence_detector = SequenceModel(**detector_config)

        classifier_dim = self.sequence_classifier.output_dim
        detector_dim = self.sequence_detector.output_dim

        self.classifier = nn.Linear(in_features=classifier_dim, out_features=self.num_class)
        self.detector = nn.Linear(in_features=detector_dim, out_features=2)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)

    def forward(self, x):
        seq_len = x.shape[-1]

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block(resblock3_out)

        classifier_out = (
            poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        classifier_out = self.sequence_classifier(classifier_out)
        classifier_out = self.classifier_dropout(classifier_out)
        classifier_out = classifier_out.contiguous().view((-1, classifier_out.shape[-1]))
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))

        mp1_out = self.maxpool1(convblock_out)
        mp2_out = self.maxpool2(resblock1_out)
        mp3_out = self.maxpool3(resblock2_out)
        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        detector_out = (
            detector_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        detector_out = self.sequence_detector(detector_out)
        detector_out = self.detector_dropout(detector_out)
        detector_out = detector_out.contiguous().view((-1, detector_out.shape[-1]))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)

        f0 = classifier_out
        if f0.dim() >= 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
        f0 = torch.abs(f0)
        return f0, detector_out, poolblock_out


def detect_jdc_variant(state_dict: Dict[str, torch.Tensor]) -> str:
    """Detect whether a checkpoint corresponds to the legacy or modern JDCNet."""

    keys = [key.replace("module.", "", 1) for key in state_dict.keys()]
    if any(key.startswith("bilstm_classifier.") for key in keys):
        return "legacy"
    return "modern"


def build_jdc_model(
    model_params: Optional[Dict] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    model_params = dict(model_params or {})
    variant = None
    if state_dict is not None:
        variant = detect_jdc_variant(state_dict)
    if variant == "legacy":
        num_class = model_params.get("num_class")
        if not isinstance(num_class, int) or num_class <= 0:
            weight = state_dict.get("classifier.weight") if state_dict else None
            if isinstance(weight, torch.Tensor):
                num_class = int(weight.shape[0])
            else:
                num_class = 1
        seq_len = model_params.get("seq_len", 31)
        return LegacyJDCNet(num_class=num_class, seq_len=seq_len)

    sequence_model_config = model_params.get("sequence_model", {})
    num_class = model_params.get("num_class")
    if not isinstance(num_class, int) or num_class <= 0:
        weight = None
        if state_dict is not None:
            weight = state_dict.get("classifier.weight")
        if isinstance(weight, torch.Tensor):
            num_class = int(weight.shape[0])
        else:
            num_class = 1
    return ModernJDCNet(
        num_class=num_class,
        sequence_model_config=sequence_model_config,
        head_dropout=model_params.get("head_dropout", 0.2),
    )


JDCNet = ModernJDCNet

__all__ = [
    "JDCNet",
    "LegacyJDCNet",
    "ModernJDCNet",
    "build_jdc_model",
    "detect_jdc_variant",
]
