"""
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import math

import torch
from torch import nn


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    def __init__(
        self,
        num_class=722,
        leaky_relu_slope=0.01,
        sequence_model_config=None,
        head_dropout=0.2,
        mel_bins=80,
    ):
        super().__init__()
        self.num_class = num_class
        self.mel_bins = int(mel_bins) if mel_bins else None
        sequence_model_config = sequence_model_config or {}
        head_dropout = float(head_dropout)
        # Constrain dropout probability to a sensible range for the heads.
        head_dropout = max(0.1, min(0.3, head_dropout))
        self.classifier_dropout = nn.Dropout(p=head_dropout)
        self.detector_dropout = nn.Dropout(p=head_dropout)

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),  # out: (b, 64, 31, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 31, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  # (b, 128, 31, 128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)  # (b, 192, 31, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 31, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.5),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        # in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        # in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        # in = (b, 640, 31, 2), out = (b, 256, 31, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )

        sequence_model_config.setdefault('input_size', 512)
        self.sequence_classifier = SequenceModel(**sequence_model_config)
        self.sequence_detector = SequenceModel(**sequence_model_config)

        classifier_dim = self.sequence_classifier.output_dim
        detector_dim = self.sequence_detector.output_dim

        # input: (b * 31, classifier_dim)
        self.classifier = nn.Linear(in_features=classifier_dim, out_features=self.num_class)  # (b * 31, num_class)

        # input: (b * 31, detector_dim)
        self.detector = nn.Linear(in_features=detector_dim, out_features=2)  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert arbitrary mel layouts to the expected 4D tensor."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                f"JDCNet expects a 3D (B, mel, T) or 4D (B, 1, mel, T) tensor, got shape {tuple(x.shape)}"
            )
        x = x.float()
        # Ensure the last dimension corresponds to the mel bins so the pooling path collapses as expected.
        if self.mel_bins is not None:
            mel_bins = self.mel_bins
            if x.shape[-1] != mel_bins and x.shape[-2] == mel_bins:
                x = x.transpose(-1, -2)
        # Fallback heuristic when mel bin count is unknown: favour the orientation that keeps the smaller
        # dimension in the last axis so very short utterances (T < mel_bins) still map the mel dimension correctly.
        if self.mel_bins is None or (x.shape[-1] != self.mel_bins and x.shape[-2] != self.mel_bins):
            if x.shape[-1] > x.shape[-2]:
                x = x.transpose(-1, -2)
        return x.contiguous()

    def _encode_stages(self, x: torch.Tensor):
        x = self._prepare_input(x)
        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)

        pool_norm = self.pool_block[0](resblock3_out)
        pool_act = self.pool_block[1](pool_norm)
        pool_reduced = self.pool_block[2](pool_act)
        poolblock_out = self.pool_block[3](pool_reduced)

        return (
            poolblock_out,
            convblock_out,
            resblock1_out,
            resblock2_out,
            resblock3_out,
            pool_act,
            pool_reduced,
        )

    def get_feature_GAN(self, x):
        poolblock_out, *_rest = self._encode_stages(x)
        pool_act = _rest[-2]
        return pool_act.transpose(-1, -2)

    def get_feature(self, x):
        poolblock_out, *_rest = self._encode_stages(x)
        pool_reduced = _rest[-1]
        return pool_reduced

    def forward(self, x):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 31, 722), (b, 31, 2)
        """
        (
            poolblock_out,
            convblock_out,
            resblock1_out,
            resblock2_out,
            _resblock3_out,
            _pool_act,
            pool_reduced,
        ) = self._encode_stages(x)
        seq_len = poolblock_out.shape[-2]
        ###############################
        # forward pass for classifier #
        ###############################
        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        classifier_out = self.sequence_classifier(classifier_out)
        classifier_out = self.classifier_dropout(classifier_out)

        classifier_out = classifier_out.contiguous().view((-1, classifier_out.shape[-1]))  # (b * 31, hidden)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))  # (b, 31, num_class)

        #############################
        # forward pass for detector #
        #############################
        mp1_out = self.maxpool1(convblock_out)
        mp2_out = self.maxpool2(resblock1_out)
        mp3_out = self.maxpool3(resblock2_out)

        # out = (b, 640, 31, 2)
        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        detector_out = detector_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        detector_out = self.sequence_detector(detector_out)  # (b, 31, hidden)
        detector_out = self.detector_dropout(detector_out)

        detector_out = detector_out.contiguous().view((-1, detector_out.shape[-1]))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)  # binary classifier - (b, seq_len)

        # sizes: (b, seq_len, 722), (b, seq_len)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of summed (isvoice, notvoice) logits per frame
        return classifier_out, detector_out, pool_reduced

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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
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
        self.register_buffer('pe', pe)

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
