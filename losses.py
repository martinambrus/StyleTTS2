import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        gen_loss = torch.mean((1-dg)**2)
        gen_losses.append(gen_loss)
        loss += gen_loss

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()
   
    
class WhisperLoss(torch.nn.Module):

    def __init__(
        self,
        model,
        wd,
        model_sr,
        slm_sr=16000,
        hop_length=300,
        freeze_encoder=True,
    ):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.whisper.parameters():
                param.requires_grad = False
        self.whisper.eval()
        self.downsample_factor = int(
            self.whisper.encoder.conv1.stride[0] * self.whisper.encoder.conv2.stride[0]
        )
        self.register_buffer(
            "base_positional_embeddings",
            self.whisper.encoder.embed_positions.weight.data.clone(),
            persistent=False,
        )

        self.wd = wd
        self.model_sr = model_sr
        self.slm_sr = slm_sr
        self.hop_length = hop_length

        feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
        self.n_fft = feature_extractor.n_fft
        self.win_length = getattr(feature_extractor, "win_length", self.n_fft)
        self.slm_hop_length = feature_extractor.hop_length
        self.register_buffer(
            "mel_filters",
            torch.tensor(feature_extractor.mel_filters, dtype=torch.float32).transpose(0, 1),
            persistent=False,
        )
        self.register_buffer(
            "hann_window", torch.hann_window(self.win_length, periodic=True), persistent=False
        )

        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def _prepare_audio(self, audio):
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)
        return audio.float()

    def _resample(self, audio):
        return self.resample.to(audio.device)(audio)

    def _target_length(self, audio):
        return max(1, int(math.ceil(audio.shape[-1] / self.hop_length)))

    def _log_mel_spectrogram(self, audio):
        window = self.hann_window.to(audio.device)
        mel_filters = self.mel_filters.to(audio.device).unsqueeze(0)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.slm_hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = stft.abs() ** 2
        mel = torch.matmul(mel_filters, magnitudes)
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))
        return log_mel

    def _encode(self, audio, target_length):
        audio = self._prepare_audio(audio)
        audio_16 = self._resample(audio)
        log_mel = self._log_mel_spectrogram(audio_16)
        seq_len = log_mel.shape[-1]
        padded_len = int(math.ceil(seq_len / self.downsample_factor) * self.downsample_factor)
        max_allowed = int(self.base_positional_embeddings.shape[0] * self.downsample_factor)
        if padded_len > max_allowed:
            padded_len = max_allowed
        if log_mel.shape[-1] > padded_len:
            log_mel = log_mel[..., :padded_len]
        elif log_mel.shape[-1] < padded_len:
            log_mel = F.pad(log_mel, (0, padded_len - log_mel.shape[-1]))
        max_positions = max(1, padded_len // self.downsample_factor)
        if self.whisper.encoder.embed_positions.num_embeddings != max_positions:
            new_embed = nn.Embedding(max_positions, self.base_positional_embeddings.shape[1])
            new_embed.weight.data.copy_(
                self.base_positional_embeddings[:max_positions].to(
                    device=log_mel.device, dtype=new_embed.weight.dtype
                )
            )
            new_embed.weight.requires_grad = False
            self.whisper.encoder.embed_positions = new_embed.to(log_mel.device)
        self.whisper.config.max_source_positions = max_positions
        log_mel = log_mel.to(self.whisper.dtype)
        outputs = self.whisper.encoder(
            input_features=log_mel, output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        processed = []
        for hs in hidden_states:
            hs = hs.transpose(1, 2)
            hs = F.interpolate(hs, size=target_length, mode="linear", align_corners=False)
            hs = hs.transpose(1, 2)
            processed.append(hs)
        stacked = torch.stack([state.transpose(1, 2) for state in processed], dim=1)
        stacked = stacked.flatten(start_dim=1, end_dim=2)
        return processed, stacked

    def forward(self, wav, y_rec):
        target_length = max(self._target_length(wav), self._target_length(y_rec))
        with torch.no_grad():
            wav_states, _ = self._encode(wav, target_length)
        y_states, _ = self._encode(y_rec, target_length)

        floss = 0.0
        for real_state, gen_state in zip(wav_states, y_states):
            floss = floss + torch.mean(torch.abs(real_state - gen_state))

        return floss / len(wav_states)

    def generator(self, y_rec):
        target_length = self._target_length(y_rec)
        _, y_embeddings = self._encode(y_rec, target_length)
        y_df_hat_g = self.wd(y_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)
        return loss_gen

    def discriminator(self, wav, y_rec):
        target_length = max(self._target_length(wav), self._target_length(y_rec))
        with torch.no_grad():
            _, y_embeddings = self._encode(wav, target_length)
            _, y_rec_embeddings = self._encode(y_rec, target_length)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        r_loss = torch.mean((1 - y_d_rs) ** 2)
        g_loss = torch.mean(y_d_gs ** 2)

        loss_disc_f = r_loss + g_loss
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        target_length = self._target_length(wav)
        with torch.no_grad():
            _, y_embeddings = self._encode(wav, target_length)
        y_d_rs = self.wd(y_embeddings)
        return y_d_rs


# Backwards compatibility with the previous name
WavLMLoss = WhisperLoss
