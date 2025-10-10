# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict
import inspect
import copy
import math

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

import logging
from contextlib import nullcontext
from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast
from torch.utils import checkpoint

try:
    from torch.amp import autocast as torch_autocast
    from torch.amp import GradScaler as TorchGradScaler
except (ImportError, AttributeError):
    torch_autocast = None
    TorchGradScaler = None
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 loss_config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 use_mixed_precision=False,
                 gradient_checkpointing=False,
                 checkpoint_use_reentrant=None,
                 ema_decay=0.0,
                 scheduler_metric='eval/vuv_f1'):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.logger = logger
        self.loss_config = loss_config or {}
        self.lambda_vuv = float(self.loss_config.get("lambda_vuv", 1.0))
        self.vuv_threshold = float(self.loss_config.get("vuv_initial_threshold", 0.5))
        self.vuv_calibration_steps = int(self.loss_config.get("vuv_calibration_steps", 201))
        if self.vuv_calibration_steps < 2:
            self.vuv_calibration_steps = 2
        if self.logger is not None:
            try:
                self.logger.info("Using lambda_vuv=%.4f for voiced/unvoiced loss", self.lambda_vuv)
                self.logger.info(
                    "Initial voiced/unvoiced threshold set to %.4f (calibration steps: %d)",
                    self.vuv_threshold,
                    self.vuv_calibration_steps,
                )
            except Exception:
                # Logger may not be fully initialised in some unit tests; fail silently.
                pass
        self.device = device
        self.finish_train = False
        device_type = torch.device(self.device).type if isinstance(self.device, (str, torch.device)) else "cpu"
        self.use_amp = bool(use_mixed_precision and device_type == "cuda")
        if TorchGradScaler is not None:
            scaler_kwargs = {"enabled": self.use_amp}
            try:
                signature = inspect.signature(TorchGradScaler.__init__)
            except (TypeError, ValueError):
                signature = None

            if signature is not None:
                parameters = signature.parameters
                if "device_type" in parameters:
                    scaler_kwargs["device_type"] = device_type
                elif "device" in parameters:
                    scaler_kwargs["device"] = device_type

            try:
                self.scaler = TorchGradScaler(**scaler_kwargs)
            except TypeError:
                scaler_kwargs.pop("device_type", None)
                scaler_kwargs.pop("device", None)
                self.scaler = TorchGradScaler(**scaler_kwargs)
            if self.use_amp:
                self.logger.info("Using mixed precision scaling with torch.amp.GradScaler")
        else:
            self.scaler = CudaGradScaler(enabled=self.use_amp)
            if self.use_amp:
                self.logger.info("Using mixed precision scaling with torch.cuda.amp.GradScaler")
        if self.use_amp:
            if torch_autocast is not None:
                def autocast_cm():
                    return torch_autocast(device_type=device_type)

                self.logger.info("Using mixed precision training with torch.amp.autocast")
            else:
                autocast_cm = cuda_autocast
                self.logger.info("Using mixed precision training with torch.cuda.amp.autocast")
        else:
            autocast_cm = nullcontext
        self._autocast_cm = autocast_cm
        self.gradient_checkpointing = bool(gradient_checkpointing and device_type == "cuda")
        if gradient_checkpointing and device_type != "cuda":
            self.logger.warning("Gradient checkpointing requested but CUDA is unavailable; disabling.")
        self._checkpoint_kwargs = {}
        self.gradient_checkpoint_use_reentrant = None
        self._checkpoint_supports_use_reentrant = False
        if self.gradient_checkpointing:
            self.logger.info("Gradient checkpointing enabled for training")
            try:
                checkpoint_signature = inspect.signature(checkpoint.checkpoint)
            except (TypeError, ValueError):
                checkpoint_signature = None

            if checkpoint_signature is not None and "use_reentrant" in checkpoint_signature.parameters:
                self._checkpoint_supports_use_reentrant = True

            desired_use_reentrant = checkpoint_use_reentrant
            if desired_use_reentrant is None and self._checkpoint_supports_use_reentrant:
                desired_use_reentrant = False

            if desired_use_reentrant is not None and not self._checkpoint_supports_use_reentrant:
                self.logger.warning(
                    "This PyTorch version does not support the use_reentrant flag; proceeding with the default checkpoint behaviour."
                )
                desired_use_reentrant = None

            self.gradient_checkpoint_use_reentrant = desired_use_reentrant

            if self.gradient_checkpoint_use_reentrant is not None:
                self.logger.info(
                    "Gradient checkpointing will run with use_reentrant=%s",
                    self.gradient_checkpoint_use_reentrant,
                )
                self._checkpoint_kwargs["use_reentrant"] = self.gradient_checkpoint_use_reentrant

        self.ema_decay = float(ema_decay) if ema_decay is not None else 0.0
        self.ema_model = None
        if self.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            self.logger.info("Model EMA enabled with decay %.6f", self.ema_decay)
        self.scheduler_metric = scheduler_metric
        self._configure_scheduler_helpers(reset_history=True)

    def save_checkpoint(self, checkpoint_path, use_ema_model=False):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            use_ema_model (bool): Whether to save EMA weights as the primary
                model state.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        else:
            state_dict["scheduler"] = None
        model_state = self.model.state_dict()
        if self.ema_model is not None:
            ema_state = self.ema_model.state_dict()
            state_dict["ema_model"] = ema_state
            state_dict["ema_decay"] = self.ema_decay
            if use_ema_model:
                state_dict["model_non_ema"] = model_state
                model_state = ema_state
        state_dict["model"] = model_state

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False, load_ema_as_model=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
            load_ema_as_model (bool): Load EMA parameters instead of raw
                weights when available.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if load_ema_as_model and "ema_model" in state_dict:
            model_key = "ema_model"
        elif not load_ema_as_model and "model_non_ema" in state_dict:
            model_key = "model_non_ema"
        else:
            model_key = "model"
        self._load(state_dict[model_key], self.model)

        self.ema_decay = state_dict.get("ema_decay", self.ema_decay)
        ema_state = state_dict.get("ema_model")
        if self.ema_decay > 0.0:
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(self.model)
                self.ema_model.to(self.device)
                for param in self.ema_model.parameters():
                    param.requires_grad_(False)
            if ema_state is not None:
                self._load(ema_state, self.ema_model)
            else:
                # Synchronize EMA weights with the loaded model when no EMA
                # snapshot is available in the checkpoint.
                self.ema_model.load_state_dict(self.model.state_dict())
                self.logger.info(
                    "EMA state missing from checkpoint; initialized EMA weights "
                    "from current model parameters."
                )

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            scheduler_state = state_dict.get("scheduler")
            if self.scheduler is not None and scheduler_state is not None:
                scheduler_state = copy.deepcopy(scheduler_state)
                scheduler_state.update(**self.config.get("scheduler_params", {}))
                self.scheduler.load_state_dict(scheduler_state)
            elif self.scheduler is not None and scheduler_state is None:
                if self.logger is not None:
                    try:
                        self.logger.warning("Scheduler state missing from checkpoint; keeping current scheduler settings.")
                    except Exception:
                        pass
            elif self.scheduler is None and scheduler_state is not None:
                if self.logger is not None:
                    try:
                        self.logger.warning("Checkpoint contains a scheduler state but no scheduler is configured; ignoring it.")
                    except Exception:
                        pass
            self._configure_scheduler_helpers(reset_history=True)

    def _configure_scheduler_helpers(self, reset_history=True):
        self.scheduler_requires_metric = isinstance(self.scheduler, ReduceLROnPlateau)

        scheduler_mode = getattr(self.scheduler, "mode", None)
        if scheduler_mode is None:
            scheduler_mode = "max"
        self._scheduler_mode = str(scheduler_mode).lower()

        scheduler_eps = float(getattr(self.scheduler, "eps", 1e-8)) if self.scheduler is not None else 1e-8
        if not math.isfinite(scheduler_eps) or scheduler_eps <= 0.0:
            scheduler_eps = 1e-8

        early_stop_config = (self.config or {}).get("early_stop", {})
        default_enabled = self.scheduler_requires_metric
        self.early_stop_enabled = bool(early_stop_config.get("enabled", default_enabled))
        self.early_stop_min_lr = float(early_stop_config.get("min_lr", 3e-6))
        if self.early_stop_min_lr < 0.0:
            self.early_stop_min_lr = 0.0
        self.early_stop_patience = max(1, int(early_stop_config.get("patience", 9)))
        self.early_stop_metric = str(early_stop_config.get("metric", self.scheduler_metric))
        self.early_stop_mode = str(early_stop_config.get("mode", self._scheduler_mode)).lower()

        configured_delta = early_stop_config.get("min_delta")
        if configured_delta is None:
            min_delta = scheduler_eps if self.early_stop_mode == self._scheduler_mode else max(1e-6, scheduler_eps)
        else:
            min_delta = max(0.0, float(configured_delta))
        self.early_stop_min_delta = min_delta

        if reset_history or not hasattr(self, "_best_early_stop_metric"):
            self._best_early_stop_metric = None
            self._epochs_since_early_stop_improve = 0

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def _update_early_stopping(self, metric_value):
        if not self.early_stop_enabled or not self.scheduler_requires_metric:
            return

        if metric_value is None:
            return

        try:
            metric_value = float(metric_value)
        except (TypeError, ValueError):
            return

        if not math.isfinite(metric_value):
            return

        if self._best_early_stop_metric is None:
            self._best_early_stop_metric = metric_value
            self._epochs_since_early_stop_improve = 0
            return

        if self.early_stop_mode == 'max':
            improved = metric_value > self._best_early_stop_metric + self.early_stop_min_delta
        else:
            improved = metric_value < self._best_early_stop_metric - self.early_stop_min_delta

        if improved:
            self._best_early_stop_metric = metric_value
            self._epochs_since_early_stop_improve = 0
            return

        self._epochs_since_early_stop_improve += 1

        current_lr = self._get_lr()
        if current_lr <= self.early_stop_min_lr and self._epochs_since_early_stop_improve >= self.early_stop_patience:
            if not self.finish_train and self.logger is not None:
                try:
                    self.logger.info(
                        "Early stopping triggered after %d stagnant epochs at learning rate %.6e (threshold %.6e) based on metric '%s'.",
                        self._epochs_since_early_stop_improve,
                        current_lr,
                        self.early_stop_min_lr,
                        self.early_stop_metric,
                    )
                except Exception:
                    pass
            self.finish_train = True

    def _update_ema(self):
        if self.ema_model is None:
            return

        with torch.no_grad():
            ema_params = list(self.ema_model.parameters())
            model_params = list(self.model.parameters())
            for ema_param, model_param in zip(ema_params, model_params):
                ema_param.mul_(self.ema_decay).add_(model_param.detach(), alpha=1.0 - self.ema_decay)

            ema_buffers = list(self.ema_model.buffers())
            model_buffers = list(self.model.buffers())
            for ema_buffer, model_buffer in zip(ema_buffers, model_buffers):
                ema_buffer.copy_(model_buffer)

    def _f0_to_loss_scale(self, f0_values):
        """Convert F0 values to the configured regression scale."""

        scale_type = str(self.loss_config.get('f0_loss_scale', 'log_hz')).lower()
        if scale_type in {'log_hz', 'log-hz', 'loghz', 'log'}:
            min_hz = float(self.loss_config.get('f0_log_min_hz', 1e-2))
            clamp_min = torch.tensor(min_hz, dtype=f0_values.dtype, device=f0_values.device)
            safe_values = torch.maximum(f0_values, clamp_min)
            return torch.log(safe_values)
        if scale_type in {'cents', 'cent'}:
            min_hz = float(self.loss_config.get('f0_log_min_hz', 1e-2))
            reference_hz = float(self.loss_config.get('f0_cents_reference_hz', 10.0))
            clamp_min = torch.tensor(min_hz, dtype=f0_values.dtype, device=f0_values.device)
            safe_values = torch.maximum(f0_values, clamp_min)
            reference = torch.tensor(reference_hz, dtype=f0_values.dtype, device=f0_values.device)
            return 1200.0 * torch.log2(safe_values / reference)
        if scale_type in {'linear_hz', 'hz', 'linear'}:
            return f0_values
        raise ValueError(f"Unsupported F0 loss scale: {scale_type}")

    def _compute_f0_loss(self, f0_pred, f0_target, silence_labels=None, voicing_probs=None, threshold=None):
        """Compute the F0 regression loss masked to voiced frames only."""

        if threshold is None:
            threshold = float(self.vuv_threshold)
        else:
            threshold = float(threshold)

        if f0_pred.dim() > f0_target.dim():
            f0_pred = f0_pred.squeeze(-1)
        if f0_target.dim() > f0_pred.dim():
            f0_target = f0_target.squeeze(-1)
        if voicing_probs is not None and voicing_probs.dim() > f0_pred.dim():
            voicing_probs = voicing_probs.squeeze(-1)
        if silence_labels is not None and silence_labels.dim() > f0_pred.dim():
            silence_labels = silence_labels.squeeze(-1)

        voiced_mask = None
        if voicing_probs is not None:
            if not isinstance(voicing_probs, torch.Tensor):
                raise TypeError("voicing_probs must be a torch.Tensor when provided")
            mask = (voicing_probs >= threshold).to(f0_pred.dtype)
            if mask.sum() < 1.0 and silence_labels is not None:
                fallback_mask = (silence_labels < 0.5).to(f0_pred.dtype)
                if fallback_mask.sum() >= 1.0:
                    mask = fallback_mask
            voiced_mask = mask
        elif silence_labels is not None:
            voiced_mask = (silence_labels < 0.5).to(f0_pred.dtype)
        else:
            voiced_mask = torch.ones_like(f0_pred, dtype=f0_pred.dtype)

        while voiced_mask.dim() < f0_pred.dim():
            voiced_mask = voiced_mask.unsqueeze(-1)

        pred_in_scale = self._f0_to_loss_scale(f0_pred)
        target_in_scale = self._f0_to_loss_scale(f0_target)
        loss_per_frame = self.criterion['f0'](pred_in_scale, target_in_scale)
        if loss_per_frame.shape != voiced_mask.shape:
            voiced_mask = voiced_mask.expand_as(loss_per_frame)
        masked_loss = loss_per_frame * voiced_mask
        normaliser = voiced_mask.sum().clamp_min(1.0)
        loss_f0 = masked_loss.sum() / normaliser
        lambda_f0 = float(self.loss_config.get('lambda_f0', 1.0))
        return lambda_f0 * loss_f0

    def _compute_vuv_metrics_from_arrays(self, voiced_probs, positive_mask, threshold):
        preds = voiced_probs >= threshold
        positive_mask = positive_mask.astype(bool)
        preds = preds.astype(bool)
        tp = np.count_nonzero(preds & positive_mask)
        fp = np.count_nonzero(preds & (~positive_mask))
        fn = np.count_nonzero((~preds) & positive_mask)
        tn = positive_mask.size - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        accuracy = (tp + tn) / positive_mask.size if positive_mask.size > 0 else 0.0

        return {
            "eval/vuv_precision": float(precision),
            "eval/vuv_recall": float(recall),
            "eval/vuv_f1": float(f1),
            "eval/vuv_accuracy": float(accuracy),
        }

    def _calibrate_vuv_threshold(self, voiced_prob_batches, silence_label_batches):
        default_metrics = {
            "eval/vuv_precision": 0.0,
            "eval/vuv_recall": 0.0,
            "eval/vuv_f1": 0.0,
            "eval/vuv_accuracy": 0.0,
        }

        if not voiced_prob_batches or not silence_label_batches:
            return None, default_metrics

        voiced_probs = torch.cat([batch.reshape(-1) for batch in voiced_prob_batches], dim=0).to(torch.float32)
        silence_labels = torch.cat([batch.reshape(-1) for batch in silence_label_batches], dim=0).to(torch.float32)

        if voiced_probs.numel() == 0:
            return None, default_metrics

        voiced_probs_np = voiced_probs.numpy()
        voiced_targets_np = (1.0 - silence_labels.numpy())

        positive_mask = voiced_targets_np >= 0.5
        if positive_mask.size == 0:
            return float(self.vuv_threshold), default_metrics.copy()

        thresholds = np.linspace(0.0, 1.0, num=self.vuv_calibration_steps)
        thresholds = np.unique(np.concatenate([thresholds, [self.vuv_threshold]]))

        best_threshold = float(self.vuv_threshold)
        best_metrics = self._compute_vuv_metrics_from_arrays(voiced_probs_np, positive_mask, best_threshold)
        best_f1 = best_metrics["eval/vuv_f1"]

        for thr in thresholds:
            metrics = self._compute_vuv_metrics_from_arrays(voiced_probs_np, positive_mask, thr)
            f1_score = metrics["eval/vuv_f1"]
            if f1_score > best_f1 + 1e-9 or (
                abs(f1_score - best_f1) <= 1e-9
                and abs(thr - self.vuv_threshold) < abs(best_threshold - self.vuv_threshold)
            ):
                best_threshold = float(thr)
                best_metrics = metrics
                best_f1 = f1_score

        return best_threshold, best_metrics

    def run(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        batch = [b.to(self.device, non_blocking=True) for b in batch]

        x, f0, sil = batch
        autocast_context = self._autocast_cm

        with autocast_context():
            if self.gradient_checkpointing:
                x = x.requires_grad_()

                def forward_fn(inp):
                    return self.model(inp.transpose(-1, -2))

                f0_pred, sil_pred = checkpoint.checkpoint(forward_fn, x, **self._checkpoint_kwargs)
            else:
                f0_pred, sil_pred = self.model(x.transpose(-1, -2))

            voicing_probs = torch.sigmoid(-sil_pred)
            loss_f0 = self._compute_f0_loss(
                f0_pred,
                f0,
                silence_labels=sil,
                voicing_probs=voicing_probs.detach(),
            )
            loss_sil_raw = self.criterion['ce'](sil_pred, sil)
            loss_sil = self.lambda_vuv * loss_sil_raw
            loss = loss_f0 + loss_sil

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self._update_ema()
        if self.scheduler is not None and not self.scheduler_requires_metric:
            self.scheduler.step()

        return {'loss': loss.item(),
                'f0': loss_f0.item(),
                'sil': loss_sil_raw.item(),
                'sil_weighted': loss_sil.item()}

    def _train_epoch(self):
        self.epochs += 1
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_model = self.ema_model if self.ema_model is not None else self.model
        base_model_training = self.model.training
        eval_model_training = eval_model.training
        eval_model.eval()
        eval_images = defaultdict(list)
        f0_batches = []
        voicing_prob_batches = []
        silence_label_batches = []
        silence_losses = []

        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device, non_blocking=True) for b in batch]
            x, f0, sil = batch

            autocast_context = self._autocast_cm
            with autocast_context():
                f0_pred, sil_pred = eval_model(x.transpose(-1, -2))

            voicing_probs = torch.sigmoid(-sil_pred)
            f0_batches.append((f0_pred.detach().cpu().float(), f0.detach().cpu().float()))
            voicing_prob_batches.append(voicing_probs.detach().cpu().float())
            silence_label_batches.append(sil.detach().cpu().float())

            loss_sil_raw = self.criterion['ce'](sil_pred, sil)
            silence_losses.append(loss_sil_raw.item())

        eval_losses = {}

        if silence_losses:
            mean_silence = float(np.mean(silence_losses))
        else:
            mean_silence = 0.0

        mean_silence_weighted = float(self.lambda_vuv * mean_silence)

        metrics = {}
        if voicing_prob_batches:
            new_threshold, metrics = self._calibrate_vuv_threshold(voicing_prob_batches, silence_label_batches)
            if new_threshold is not None and not np.isclose(new_threshold, self.vuv_threshold):
                previous_threshold = self.vuv_threshold
                self.vuv_threshold = float(new_threshold)
                if self.logger is not None:
                    try:
                        self.logger.info(
                            "Updated V/UV threshold from %.4f to %.4f after calibration",
                            previous_threshold,
                            self.vuv_threshold,
                        )
                    except Exception:
                        pass
        else:
            metrics = {
                "eval/vuv_f1": 0.0,
                "eval/vuv_precision": 0.0,
                "eval/vuv_recall": 0.0,
                "eval/vuv_accuracy": 0.0,
            }

        metrics["eval/vuv_threshold"] = float(self.vuv_threshold)

        f0_losses = []
        for (f0_pred_cpu, f0_target_cpu), probs_cpu, sil_cpu in zip(f0_batches, voicing_prob_batches, silence_label_batches):
            loss_value = self._compute_f0_loss(
                f0_pred_cpu,
                f0_target_cpu,
                silence_labels=sil_cpu,
                voicing_probs=probs_cpu,
                threshold=self.vuv_threshold,
            )
            f0_losses.append(loss_value.item())

        if f0_losses:
            mean_f0 = float(np.mean(f0_losses))
        else:
            mean_f0 = 0.0

        total_loss = float(mean_f0 + mean_silence_weighted)

        eval_losses["eval/loss"] = total_loss
        eval_losses["eval/f0"] = mean_f0
        eval_losses["eval/sil"] = mean_silence
        eval_losses["eval/sil_weighted"] = mean_silence_weighted
        eval_losses.update(metrics)
        eval_losses.update(eval_images)
        eval_model.train(eval_model_training)
        self.model.train(base_model_training)
        return eval_losses

    def update_scheduler(self, metrics=None):
        if self.scheduler is None:
            self._update_early_stopping(None)
            return

        metric_value = None
        if self.scheduler_requires_metric:
            if isinstance(metrics, dict):
                metric_value = metrics.get(self.scheduler_metric)
                if metric_value is None and self.scheduler_metric != 'eval/loss':
                    metric_value = metrics.get('eval/loss')

            if metric_value is None:
                if self.logger is not None:
                    try:
                        self.logger.warning(
                            "Unable to find metric '%s' for scheduler update; skipping step.",
                            self.scheduler_metric,
                        )
                    except Exception:
                        pass
                self._update_early_stopping(None)
                return

            self.scheduler.step(metric_value)

        early_stop_value = None
        if isinstance(metrics, dict):
            early_stop_value = metrics.get(self.early_stop_metric)
            if early_stop_value is None and self.early_stop_metric == 'eval/loss':
                early_stop_value = metrics.get('eval/loss')
            if early_stop_value is None and self.early_stop_metric == self.scheduler_metric:
                early_stop_value = metric_value
        elif self.early_stop_metric == self.scheduler_metric:
            early_stop_value = metric_value

        self._update_early_stopping(early_stop_value)
