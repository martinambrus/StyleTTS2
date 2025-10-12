import torch
import torch.nn.functional as F
import torchaudio
from contextlib import nullcontext
from typing import Tuple
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel

from Modules.whisper_processor import DifferentiableWhisperFeatureExtractor

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
   
    
class WavLMLoss(torch.nn.Module):

    def __init__(self, model, wd, model_sr, slm_sr=16000, slm_type="wavlm"):
        super(WavLMLoss, self).__init__()
        self.slm_type = slm_type.lower()
        if self.slm_type == "wavlm":
            self.slm = AutoModel.from_pretrained(model)
            self.feature_extractor = None
        elif self.slm_type == "whisper":
            whisper_model = WhisperModel.from_pretrained(model)
            self.slm = whisper_model.encoder
            feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
            self.feature_extractor = DifferentiableWhisperFeatureExtractor(feature_extractor)
        else:
            raise ValueError(f"Unsupported SLM type: {slm_type}")

        self.slm.eval()
        for param in self.slm.parameters():
            param.requires_grad_(False)

        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.slm_sr = slm_sr

    def _prepare_resampled(self, audio: torch.Tensor) -> torch.Tensor:
        audio_16 = self.resample(audio)
        if audio_16.dim() == 3 and audio_16.size(1) == 1:
            audio_16 = audio_16.squeeze(1)
        if audio_16.dim() == 1:
            audio_16 = audio_16.unsqueeze(0)
        return audio_16

    def _encode(self, audio: torch.Tensor, require_grad: bool) -> Tuple[torch.Tensor, ...]:
        audio_16 = self._prepare_resampled(audio)
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            if self.slm_type == "wavlm":
                # Some WavLM checkpoints (e.g. torchaudio implementations) expect
                # the waveform as the first positional argument instead of the
                # HuggingFace-style ``input_values`` keyword. Passing the tensor
                # positionally keeps compatibility with both variants.
                outputs = self.slm(audio_16, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            else:
                features = self.feature_extractor(audio_16, sampling_rate=self.slm_sr)["input_features"]
                outputs = self.slm(input_features=features, output_hidden_states=True)
                hidden_states = outputs.hidden_states
        return hidden_states

    @staticmethod
    def _stack_hidden_states(hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=1).transpose(-1, -2)
        return stacked.flatten(start_dim=1, end_dim=2).contiguous()

    def forward(self, wav, y_rec):
        wav_embeddings = self._encode(wav, require_grad=False)
        y_rec_embeddings = self._encode(y_rec, require_grad=True)

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        y_rec_embeddings = self._encode(y_rec, require_grad=True)
        y_rec_embeddings = self._stack_hidden_states(y_rec_embeddings)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        wav_embeddings = self._encode(wav, require_grad=False)
        y_rec_embeddings = self._encode(y_rec, require_grad=False)

        y_embeddings = self._stack_hidden_states(wav_embeddings)
        y_rec_embeddings = self._stack_hidden_states(y_rec_embeddings)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        r_loss = torch.mean((1 - y_d_rs) ** 2)
        g_loss = torch.mean(y_d_gs ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        wav_embeddings = self._encode(wav, require_grad=False)
        y_embeddings = self._stack_hidden_states(wav_embeddings)

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
