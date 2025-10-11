from typing import Optional, Union
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import torch
import importlib

def _load_whisper():
    global whisper
    if 'whisper' not in globals():
        whisper = importlib.import_module('whisper')

class DifferentiableWhisperFeatureExtractor(WhisperFeatureExtractor):
  def __init__(self, wfe:WhisperFeatureExtractor):
    self.hop_length=wfe.hop_length
    self.n_fft=wfe.n_fft
    self.dither=getattr(wfe, "dither", 0.0)
    self.sampling_rate=wfe.sampling_rate
    self.feature_size=wfe.feature_size
    self.mel_filters = wfe.mel_filters
    self.n_samples = wfe.n_samples
    self.return_attention_mask = wfe.return_attention_mask
    self.padding_side = wfe.padding_side
    self.padding_value = wfe.padding_value


  def _hf_differentiable_extract_fbank_features(self, waveform, device=None, dtype=None):
    """Return differentiable log-Mel spectrogram on the specified device and dtype."""
    _load_whisper()
    if device is None:
        device = waveform.device
    if dtype is None:
        dtype = waveform.dtype

    # whisper's helper already performs the STFT and Mel conversion
    log_spec = whisper.audio.log_mel_spectrogram(
        waveform.to(device=device, dtype=dtype), n_mels=self.feature_size, device=device
    )
    return log_spec.to(device=device, dtype=dtype)




  def __call__(
      self,
      raw_speech: torch.Tensor,
      truncation: bool = True,
      pad_to_multiple_of: Optional[int] = None,
      return_tensors: Optional[Union[str, TensorType]] = None,
      return_attention_mask: Optional[bool] = None,
      padding: Optional[str] = "max_length",
      max_length: Optional[int] = None,
      sampling_rate: Optional[int] = None,
      do_normalize: Optional[bool] = None,
      device: Optional[Union[str, torch.device]] = None,
      dtype: Optional[torch.dtype] = None,
      return_token_timestamps: Optional[bool] = None,
      **kwargs,
  ) -> BatchFeature:
      """
      New diff
      Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
      the STFT computation if available, otherwise a slower NumPy based one.

      Args:
          raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
              The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
              values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
              stereo, i.e. single float per timestep.
          truncation (`bool`, *optional*, default to `True`):
              Activates truncation to cut input sequences longer than *max_length* to *max_length*.
          pad_to_multiple_of (`int`, *optional*, defaults to None):
              If set will pad the sequence to a multiple of the provided value.

              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
              `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
          return_attention_mask (`bool`, *optional*):
              Whether to return the attention mask. If left to the default, will return the attention mask according
              to the specific feature_extractor's default.

              [What are attention masks?](../glossary#attention-mask)

              <Tip>

              For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
              bugs.

              </Tip>

          return_tensors (`str` or [`~utils.TensorType`], *optional*):
              If set, will return tensors instead of list of python integers. Acceptable values are:

              - `'tf'`: Return TensorFlow `tf.constant` objects.
              - `'pt'`: Return PyTorch `torch.Tensor` objects.
              - `'np'`: Return Numpy `np.ndarray` objects.
          sampling_rate (`int`, *optional*):
              The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
              `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
              pipeline.
          padding_value (`float`, *optional*, defaults to 0.0):
              The value that is used to fill the padding values / vectors.
          do_normalize (`bool`, *optional*, defaults to `False`):
              Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
              improve the performance of the model.
          device (`str` or `torch.device`, *optional*):
              Device used when computing the log-mel spectrogram. Defaults to the device of
              `raw_speech` if not provided.
          dtype (`torch.dtype`, *optional*):
              Desired floating point type of the returned features. Defaults to the dtype of
              `raw_speech`.
          return_token_timestamps (`bool`, *optional*, defaults to `None`):
              Whether or not to return the number of frames of the input raw_speech.
              These num_frames can be used by the model to compute word level timestamps.
      """

      # if raw_speech.ndim == 3 and raw_speech.size(1) == 1:
      #   raw_speech = raw_speech.squeeze(1)
      # elif raw_speech.ndim == 1:
      #   raw_speech = raw_speech.unsqueeze(0)  # ensure batch dim
      # elif raw_speech.ndim != 2:
      #   raise ValueError(f"Unexpected raw_speech shape: {raw_speech.shape}")

      

      # print("padding", padding)
      # print("truncation", truncation)
      # print("pad_to_multiple_of", pad_to_multiple_of)
      # print("return_attention_mask", return_attention_mask)
      # print("do_normalize", do_normalize)

      max_length = max_length if max_length else self.n_samples
      current_max_len = raw_speech.shape[-1]

      if current_max_len > max_length:
         raw_speech = raw_speech[..., 0:max_length]
      elif current_max_len < max_length:
        batch_size = raw_speech.shape[0]
        new_len = max_length - current_max_len
        zeros = torch.zeros((batch_size, new_len), device=raw_speech.device)
        raw_speech = torch.concat([raw_speech, zeros], dim=1)

      # print("raw_speech grads", raw_speech.requires_grad)

      input_features = self._hf_differentiable_extract_fbank_features(
          raw_speech, device, dtype
      )

      res = BatchFeature({
          "input_features": input_features
      })

      return res
