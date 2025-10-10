"""Enhanced JDCNet implementation compatible with legacy checkpoints."""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class JDCNet(nn.Module):
    """Joint Detection and Classification Network model for singing voice melody."""

    def __init__(
        self,
        num_class: int = 722,
        leaky_relu_slope: float = 0.01,
        sequence_model_config: Optional[Dict] = None,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_class = int(num_class)
        sequence_model_config = dict(sequence_model_config or {})
        head_dropout = max(0.1, min(0.3, float(head_dropout)))
        self.classifier_dropout = nn.Dropout(p=head_dropout)
        self.detector_dropout = nn.Dropout(p=head_dropout)

        # input = (b, 1, 31, 513)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )

        self.res_block1 = ResBlock(64, 128, leaky_relu_slope=leaky_relu_slope)
        self.res_block2 = ResBlock(128, 192, leaky_relu_slope=leaky_relu_slope)
        self.res_block3 = ResBlock(192, 256, leaky_relu_slope=leaky_relu_slope)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(256),
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

        sequence_model_config.setdefault("input_size", 512)
        self.sequence_classifier = SequenceModel(**sequence_model_config)
        self.sequence_detector = SequenceModel(**sequence_model_config)

        classifier_dim = self.sequence_classifier.output_dim
        detector_dim = self.sequence_detector.output_dim

        self.classifier = nn.Linear(classifier_dim, self.num_class)
        self.detector = nn.Linear(detector_dim, 2)

        self.apply(self.init_weights)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pitch logits, intermediate GAN features and pooled features."""

        seq_len = x.shape[-2]
        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        pool_bn = self.pool_block[0](resblock3_out)
        pool_act = self.pool_block[1](pool_bn)
        pool_mp = self.pool_block[2](pool_act)
        gan_feature = pool_mp.transpose(-1, -2)
        poolblock_out = self.pool_block[3](pool_mp)

        classifier_out = (
            poolblock_out.permute(0, 2, 1, 3)
            .contiguous()
            .view((-1, seq_len, 512))
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
            detector_out.permute(0, 2, 1, 3)
            .contiguous()
            .view((-1, seq_len, 512))
        )
        detector_out = self.sequence_detector(detector_out)
        detector_out = self.detector_dropout(detector_out)
        detector_out = detector_out.contiguous().view((-1, detector_out.shape[-1]))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)

        pitch_logits = torch.abs(classifier_out)
        return pitch_logits, gan_feature, poolblock_out

    def extract_pitch(self, x: torch.Tensor) -> torch.Tensor:
        pitch, _, _ = self.forward(x)
        return pitch.squeeze(-1)

    def get_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pitch, _, pooled = self.forward(x)
        return pitch.squeeze(-1), pooled

    def get_feature_GAN(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, gan_feature, pooled = self.forward(x)
        return gan_feature, pooled

    @staticmethod
    def init_weights(m: nn.Module) -> None:
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
                if p.data.ndim >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))

        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2)),
            )
        else:
            self.downsample_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_layer(residual)

        out += residual
        out = self.leaky_relu(out)
        out = self.maxpool(out)
        return out


class SinusoidalPositionalEncoding(nn.Module):
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
    ) -> None:
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


__all__ = ["JDCNet", "ResBlock", "SequenceModel"]
