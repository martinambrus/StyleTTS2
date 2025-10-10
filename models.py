#coding:utf-8
import math

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)

from munch import Munch
import yaml


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 1),
                stride=(2, 1),
                groups=dim_in,
                output_padding=(1, 0),
                padding=(1, 0),
            )
        elif self.layer_type == "half":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 3),
                stride=(2, 2),
                groups=dim_in,
                output_padding=1,
                padding=1,
            )
        else:
            raise RuntimeError(
                "Got unexpected upsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        elif self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise RuntimeError(
                "Got unexpected upsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)

        return s


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            features.append(x)
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
        dropout_p=0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.Conv1d(
                    dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1
                )
            )

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == "none":
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()

        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )

        self.lstm = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.F0.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.N.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False
        )

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, : x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(
            nn.functional.dropout(x, 0.5, training=self.training)
        )

        en = d.transpose(-1, -2) @ alignment

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)

        return x.transpose(-1, -2)

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


def _deep_merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for data in dicts:
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _deep_merge_dicts(merged[key], value)
            elif isinstance(value, dict):
                merged[key] = _deep_merge_dicts(value)
            else:
                merged[key] = value
    return merged


def _resolve_path(base_dir: Optional[Path], candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_file() and base_dir is not None:
        path = (base_dir / candidate).expanduser()
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def load_F0_models(path, config_path: Optional[str] = None):
    """Load an F0 model trained with the legacy or enhanced PitchExtractor repos."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict: Dict[str, torch.Tensor]
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("net")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected checkpoint format for F0 model")

    config_sources = []
    if config_path:
        cfg_file = _resolve_path(Path(config_path).parent, config_path)
        if cfg_file is None:
            cfg_file = Path(config_path)
        try:
            with open(cfg_file) as f:
                cfg_data = yaml.safe_load(f)
        except FileNotFoundError:
            cfg_data = None
        if isinstance(cfg_data, dict):
            config_sources.append(cfg_data.get("model_params"))
    if isinstance(checkpoint, dict):
        config_sources.append(checkpoint.get("model_params"))
        config_section = checkpoint.get("config")
        if isinstance(config_section, dict):
            config_sources.append(config_section.get("model_params"))

    model_params = _deep_merge_dicts(*config_sources)
    sequence_model_config = {}
    if isinstance(model_params, dict):
        sequence_model_config = model_params.pop("sequence_model", {}) or {}
    else:
        model_params = {}

    classifier_weight = state_dict.get("classifier.weight")
    inferred_classes = None
    if isinstance(classifier_weight, torch.Tensor) and classifier_weight.ndim >= 1:
        inferred_classes = int(classifier_weight.shape[0])

    num_class = model_params.pop("num_class", None)
    if not isinstance(num_class, int) or num_class <= 0:
        num_class = inferred_classes if inferred_classes is not None else 1

    kwargs: Dict[str, Any] = {
        "num_class": num_class,
    }
    for key in ["leaky_relu_slope", "head_dropout"]:
        value = model_params.pop(key, None)
        if isinstance(value, (int, float)):
            kwargs[key] = float(value)
    if sequence_model_config:
        kwargs["sequence_model_config"] = sequence_model_config

    F0_model = JDCNet(**kwargs)
    missing = F0_model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        # Ignore incompatible keys silently to retain backwards compatibility.
        pass
    _ = F0_model.train()

    return F0_model


def _load_token_map(config: Dict[str, Any], config_path: Path) -> Optional[Dict[str, int]]:
    token_src = config.get("phoneme_maps_path")
    if isinstance(token_src, dict):
        try:
            return {str(k): int(v) for k, v in token_src.items()}
        except (TypeError, ValueError):
            return None

    path = None
    if isinstance(token_src, str):
        path = _resolve_path(config_path.parent, token_src) or Path(token_src)
    if path is not None and path.is_file():
        token_map: Dict[str, int] = {}
        try:
            with open(path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    symbol = row[0].strip()
                    if symbol.startswith("\"") and symbol.endswith("\"") and len(symbol) >= 2:
                        symbol = symbol[1:-1]
                    try:
                        token_map[symbol] = int(row[1])
                    except ValueError:
                        continue
        except OSError:
            token_map = {}
        if token_map:
            return token_map

    return None


def _infer_n_token_from_state_dict(state_dict: Dict[str, Any]) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None

    candidate_sizes = []

    def _record_size(tensor: Any) -> None:
        if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
            size = int(tensor.size(0))
            if size > 0:
                candidate_sizes.append(size)

    known_suffixes = (
        "asr_s2s.embedding.weight",
        "asr_s2s.project_to_n_symbols.weight",
        "asr_s2s.project_to_n_symbols.bias",
        "ctc_classifier.linear_layer.weight",
        "ctc_classifier.linear_layer.bias",
        "duration_predictor.0.weight",
        "frame_classifier.2.linear_layer.weight",
        "frame_classifier.2.linear_layer.bias",
    )

    sc_block_markers = (
        ".predictor.6.conv.weight",
        ".predictor.6.conv.bias",
        ".condition_projector.1.conv.weight",
    )

    ictc_markers = (
        ".layers.3.conv.weight",
        ".layers.3.conv.bias",
    )

    for raw_key, tensor in state_dict.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key

        for suffix in known_suffixes:
            if key.endswith(suffix):
                _record_size(tensor)
                break
        else:
            if "self_conditioning_blocks" in key:
                if any(key.endswith(marker) for marker in sc_block_markers):
                    _record_size(tensor)
            elif "intermediate_ctc_heads" in key:
                if any(key.endswith(marker) for marker in ictc_markers):
                    _record_size(tensor)

    if not candidate_sizes:
        return None

    counts = Counter(candidate_sizes)
    most_common, _ = counts.most_common(1)[0]
    return int(most_common)


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model compatible with legacy and enhanced AuxiliaryASR repos
    config_path = Path(ASR_MODEL_CONFIG)
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    model_config = {}
    if isinstance(raw_config, dict):
        model_config = dict(raw_config.get("model_params", {}) or {})

    token_map = _load_token_map(raw_config or {}, config_path)
    n_token = model_config.get("n_token")
    if (not isinstance(n_token, int) or n_token <= 0) and token_map is not None:
        model_config["n_token"] = len(token_map)

    stabilization_cfg = raw_config.get("stabilization") if isinstance(raw_config, dict) else None
    if isinstance(stabilization_cfg, dict):
        model_config.setdefault("stabilization_config", stabilization_cfg)

    multi_task_cfg = raw_config.get("multi_task") if isinstance(raw_config, dict) else None
    if isinstance(multi_task_cfg, dict):
        model_config.setdefault("multi_task_config", multi_task_cfg)

    memory_opt_cfg = raw_config.get("memory_optimizations") if isinstance(raw_config, dict) else None
    if isinstance(memory_opt_cfg, dict):
        model_config.setdefault("memory_optimization_config", memory_opt_cfg)

    def _load_model(model_params: Dict[str, Any], model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model")
                or checkpoint.get("ema_model")
                or checkpoint.get("model_non_ema")
                or checkpoint.get("state_dict")
            )
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError("Unexpected checkpoint format for ASR model")
        cleaned_state = {}
        for key, value in state_dict.items():
            if isinstance(key, str) and key.startswith("module."):
                cleaned_state[key[7:]] = value
            else:
                cleaned_state[key] = value

        inferred_n_token = _infer_n_token_from_state_dict(cleaned_state)
        if isinstance(inferred_n_token, int) and inferred_n_token > 0:
            configured_tokens = model_params.get("n_token")
            if not isinstance(configured_tokens, int) or configured_tokens <= 0:
                model_params["n_token"] = inferred_n_token
            elif configured_tokens != inferred_n_token:
                model_params["n_token"] = inferred_n_token

        model = ASRCNN(**model_params)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(cleaned_state, strict=False)
        return model

    asr_model = _load_model(model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model


def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ["istftnet", "hifigan"], "Decoder type unknown"

    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder

        decoder = Decoder(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=args.decoder.gen_istft_n_fft,
            gen_istft_hop_size=args.decoder.gen_istft_hop_size,
        )
    else:
        from Modules.hifigan import Decoder

        decoder = Decoder(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
        )

    text_encoder = TextEncoder(
        channels=args.hidden_dim,
        kernel_size=5,
        depth=args.n_layer,
        n_symbols=args.n_token,
    )

    predictor = ProsodyPredictor(
        style_dim=args.style_dim,
        d_hid=args.hidden_dim,
        nlayers=args.n_layer,
        max_dur=args.max_dur,
        dropout=args.dropout,
    )

    style_encoder = StyleEncoder(
        dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
    )  # acoustic style encoder
    predictor_encoder = StyleEncoder(
        dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
    )  # prosodic style encoder

    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            context_features=args.style_dim * 2,
            **args.diffusion.transformer
        )
    else:
        transformer = Transformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            **args.diffusion.transformer
        )

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,  # Conditional dropout of batch elements,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )

    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(
            mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
        ),
        sigma_data=args.diffusion.dist.sigma_data,  # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0,
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    nets = Munch(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),
        predictor=predictor,
        decoder=decoder,
        text_encoder=text_encoder,
        predictor_encoder=predictor_encoder,
        style_encoder=style_encoder,
        diffusion=diffusion,
        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,
        mpd=MultiPeriodDiscriminator(),
        msd=MultiResSpecDiscriminator(),
        # slm discriminator head
        wd=WavLMDiscriminator(
            args.slm.hidden, args.slm.nlayers, args.slm.initial_channel
        ),
    )

    return nets


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                print(f'{key} key length: {len(model[key].state_dict().keys())}, state_dict key length: {len(state_dict.keys())}')
                for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
            print("%s loaded" % key)

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
