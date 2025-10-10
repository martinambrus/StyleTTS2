# JDC-PitchExtractor
This repo contains the training code for deep neural pitch extractor for Voice Conversion (VC) and TTS used in [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC) and [StyleTTS](https://github.com/yl4579/StyleTTS). This is the F0 network in StarGANv2-VC and pitch extractor in StyleTTS. 

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/PitchExtractor.git
cd PitchExtractor
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio torch pyyaml click matplotlib librosa pyworld resampy torchcrepe tensorflow
```
4. (Optional) Install extra dependencies for alternative F0 backends:

   | Backend        | Installation command / notes |
   | -------------- | ----------------------------- |
   | TorchCrepe     | `pip install torchcrepe` |
   | SwiftF0        | `pip install swift-f0` (installs ONNX Runtime; requires `librosa` when resampling) |
   | Praat / Parselmouth | `pip install praat-parselmouth` |

   These packages are only needed when you enable the corresponding backend in `Configs/config.yml`. The training pipeline will gracefully skip any backend whose dependency is missing.

5. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training
```bash
python train.py --config_path ./Configs/config.yml
```
Please specify the training and validation data in `config.yml` file. The data list format needs to be `filename.wav|anything`, see [train_list.txt](https://github.com/yl4579/StarGANv2-VC/blob/main/Data/train_list.txt) as an example (a subset of VCTK). Note that you can put anything after the filename because the training labels are generated ad-hoc.

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take.

During validation the trainer now automatically calibrates the voiced/unvoiced
decision threshold by sweeping possible values on the held-out set and picking
the one that maximises the F1 score. The resulting threshold is reused for the
F0 loss mask and the reported V/UV metrics so evaluation mirrors the behaviour
seen at inference time. You can override the initial guess (`vuv_initial_threshold`)
and the resolution of the sweep (`vuv_calibration_steps`) from the `loss_params`
section of [`Configs/config.yml`](Configs/config.yml) when needed.

### Loss weighting sweeps
The total objective is formulated as `L_total = L_f0 + λ · L_vuv`, where `L_vuv`
is the voiced/unvoiced (silence) classification loss. You can explore different
values of `λ` by editing the `loss_params` block in
[`Configs/config.yml`](Configs/config.yml). When
`lambda_vuv_candidates` contains more than one value the training script will
automatically launch one experiment per candidate, saving checkpoints under
`<log_dir>/lambda_vuv_<value>`. The default configuration tries `λ ∈ {0.25, 0.5,
1.0}` with a base weight of `lambda_vuv = 0.5`. Remove or empty the candidate
list if you only wish to run a single setting.

### Mixed precision and memory optimisations
When running on CUDA the trainer now enables automatic mixed precision (AMP) and gradient scaling by default, significantly reducing VRAM usage while taking advantage of Tensor Cores. You can control this behaviour via the `training` section of [`Configs/config.yml`](Configs/config.yml): set `mixed_precision` to `false` to force full FP32 training or flip `gradient_checkpointing` to `true` if you want to trade a small amount of compute for lower activation memory during the forward pass. When checkpointing is active you can also decide whether to use the legacy re-entrant autograd path by toggling `gradient_checkpointing_use_reentrant` (defaults to the recommended `false` on recent PyTorch releases and automatically falls back when the flag is unsupported).

### Sequence modelling options
The default configuration now employs a Transformer encoder on top of the convolutional stack to provide stronger long-term temporal context and reduce octave jumps. You can switch between a deeper bidirectional LSTM and the Transformer backend by editing `model_params.sequence_model` in [Configs/config.yml](Configs/config.yml). The section exposes typical hyper-parameters (number of layers, attention heads, feed-forward width, etc.) so you can tailor the temporal model to your dataset.

### IMPORTANT: DATA FOLDER NEEDS WRITE PERMISSION
Since both `harvest` and `dio` are relatively slow, we do have to save the computed F0 ground truth for later use. In [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py#L77-L89), it will write the computed F0 curve `_f0.npy` for each `.wav` file. This requires write permission in your data folder.

### F0 Computation Details
`meldataset.MelDataset` now supports a cascade of runtime-selectable F0 backends. The default configuration uses PyWorld's `harvest` followed by `dio`, mirroring the original behaviour, but you can enable neural or classical trackers such as TorchCrepe (PyTorch CREPE) and Praat/Parselmouth by editing `dataset_params.f0_params` in [Configs/config.yml](Configs/config.yml). Backends are evaluated in the order defined by `backend_order`, and each backend may be toggled on/off or customised individually (for example, to adjust TorchCrepe's model size or Praat's pitch range).

Whenever the backend configuration changes the dataset automatically regenerates cached pitch files under backend-specific filenames and stores a small JSON metadata file alongside each cache to keep track of the extractor that produced it. Failed extraction attempts fall back to the next enabled backend until a valid contour with sufficient voiced frames is produced. If all backends fail the sample is logged and the stored F0 is left empty (zeros after post-processing), so you may want to audit those cases if they occur frequently.

#### Backend configuration summary

- **PyWorld (harvest/dio/stonemask)** – Controlled by the `algorithm`, optional `fallback` algorithm, and `stonemask` refinement flag.
- **TorchCrepe (CREPE)** – Choose `model` (`tiny`, `small`, `medium`, `full`), override `step_size_ms`, constrain the search range via `fmin`/`fmax`, and tune batching (`batch_size`), padding, and optional `median_filter_size`. Enable `return_periodicity` to obtain confidence scores and zero out low-confidence frames with `periodicity_threshold`. `device` accepts `auto` (prefer CUDA) or an explicit torch device string such as `cpu`, `cuda`, or `cuda:1`. When any enabled backend requests CUDA the dataloader automatically switches to the `spawn` multiprocessing context so TorchCrepe can initialise GPUs inside worker processes; override this behaviour via `dataset_params.dataloader.start_method` if necessary.
- **SwiftF0** – Uses a learnable filterbank front-end and an ONNX Runtime CNN to analyse spectrogram patches. Configure `confidence_threshold`, `fmin`, `fmax`, and `unvoiced_value` to focus on the frequency band that matters for your data, suppress noisy detections, and control how unvoiced frames are filled. Runs efficiently on CPU; requires the `swift-f0` package (and `librosa` for resampling when your dataset is not at 16 kHz).
- **Praat / Parselmouth** – Set the `method` (e.g., `ac`, `cc`), `min_pitch`/`max_pitch` bounds, and adjust `silence_threshold` and `voicing_threshold` for sensitivity.

The optional `dataset_params.dataloader` dictionary lets you fine-tune how the `DataLoader` is constructed (for example, setting `start_method`, `persistent_workers`, or `prefetch_factor`). When omitted the builder keeps PyTorch defaults, only forcing `start_method: spawn` when CUDA-enabled F0 backends would otherwise hit the "Cannot re-initialize CUDA in forked subprocess" error.

### Data Augmentation
Data augmentation is not included in this code. For better voice conversion results, please add your own data augmentation in [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py) with [audiomentations](https://github.com/iver56/audiomentations).

### Synthetic pitch-calibrated data
The dataloader can optionally blend in synthetic utterances whose pitch is known
exactly. Enable the `dataset_params.synthetic_data` section in
[Configs/config.yml](Configs/config.yml) to either pitch-shift existing clips or
render fresh vowel-like speech with the WORLD vocoder. Synthetic data is
especially useful for calibrating the regression head: even a small proportion
of perfectly labelled samples nudges the model towards the correct absolute
frequency scale (a trick popularised by Google's SwiftF0 training recipe).

Two strategies are available:

1. **Pitch shifting real audio** – Requires `librosa`. The loader draws random
   files from the training list, applies semitone offsets, and scales the
   cached F0 trajectory analytically. Adjust the `semitones`,
   `gain_db_range`, and `min_voiced_fraction` parameters to control how
   aggressive the augmentation should be.
2. **WORLD vocoder synthesis** – Requires `pyworld`. The synthesiser generates
   short vowel snippets with controllable duration, pitch range, vibrato, and
   energy. Because the excitation contour is specified up-front, the ground
   truth F0 is exact. Tune the `duration`, `pitch_range`, `gain_db_range`, and
   `modulation` blocks to match your dataset.

Synthetic examples are only added to the training split by default. Set
`apply_to_validation: true` if you also want the validation loader to include
them (for instance, when building dedicated calibration subsets).

## References
- [keums/melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
