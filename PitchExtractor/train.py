from model import JDCNet
from meldataset import build_dataloader
from optimizers import build_optimizer
from trainer import Trainer
from losses import build_silence_loss

import time
import os
import os.path as osp
import re
import sys
import yaml
import shutil
import copy
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
import click
from tqdm import tqdm

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    # train_list = train_list[-500:]
    # val_list = train_list[:500]
    return train_list, val_list

def _train_single_run(config, config_path, run_log_dir):
    os.makedirs(run_log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(run_log_dir, osp.basename(config_path)))

    writer = SummaryWriter(run_log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(run_log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 32)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 100)
    save_freq = config.get('save_freq', 10)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    num_workers = config.get('num_workers', 8)
    training_config = config.get('training', {})

    train_list, val_list = get_data_path_list(train_path, val_path)

    dataset_config = config.get('dataset_params', {})
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        dataset_config=dataset_config,
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=max(1, num_workers // 2),
                                      device=device,
                                      dataset_config=dataset_config)

    # define model
    model_config = config.get('model_params', {})
    sequence_model_config = model_config.get('sequence_model', {})
    model = JDCNet(
        num_class=model_config.get('num_class', 1),  # num_class = 1 means regression
        sequence_model_config=sequence_model_config,
        head_dropout=float(model_config.get('head_dropout', 0.2)),
    )

    optimizer_params = config.get('optimizer_params', {})
    scheduler_type = str(optimizer_params.get('scheduler_type', 'one_cycle')).lower()
    scheduler_params = {
        "type": scheduler_type,
        "max_lr": float(optimizer_params.get('lr', 5e-4)),
        "pct_start": float(optimizer_params.get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    if scheduler_type in {'reduce_on_plateau', 'plateau'}:
        min_lr = float(optimizer_params.get('plateau_min_lr', 1e-6))
        if min_lr < 0.0:
            min_lr = 0.0
        scheduler_params.update({
            "mode": str(optimizer_params.get('plateau_mode', 'max')),
            "factor": float(optimizer_params.get('plateau_factor', 0.5)),
            "patience": int(optimizer_params.get('plateau_patience', 3)),
            "threshold": float(optimizer_params.get('plateau_threshold', 1e-4)),
            "threshold_mode": str(optimizer_params.get('plateau_threshold_mode', 'rel')),
            "cooldown": int(optimizer_params.get('plateau_cooldown', 1)),
            "min_lr": min_lr,
            "eps": float(optimizer_params.get('plateau_eps', 1e-8)),
        })

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params": optimizer_params, "scheduler_params": scheduler_params})

    loss_config = config.get('loss_params', {})
    f0_delta = float(loss_config.get('f0_huber_delta', 0.075))

    criterion = {
        'f0': nn.SmoothL1Loss(reduction='none', beta=f0_delta),  # F0 loss (regression)
        'ce': build_silence_loss(loss_config)  # silence loss (binary classification)
    }

    trainer_config = {'scheduler_params': scheduler_params,
                      'early_stop': copy.deepcopy(training_config.get('early_stop', {}))}

    trainer = Trainer(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=trainer_config,
                        device=device,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        loss_config=loss_config,
                        logger=logger,
                        use_mixed_precision=training_config.get('mixed_precision', True),
                        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
                        checkpoint_use_reentrant=training_config.get('gradient_checkpointing_use_reentrant'),
                        ema_decay=training_config.get('ema_decay', 0.0),
                        scheduler_metric=training_config.get('scheduler_metric', 'eval/vuv_f1'))

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    # compute all F0 for training and validation data
    print('Checking if all F0 data is computed...')
    for _ in enumerate(train_dataloader):
        continue
    for _ in enumerate(val_dataloader):
        continue
    print('All F0 data is computed.')

    best_eval_loss = float('inf')
    best_eval_f1 = float('-inf')
    best_eval_f0 = float('inf')
    metric_epsilon = 1e-9
    for epoch in range(1, epochs+1):
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        trainer.update_scheduler(eval_results)
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.4f' % (key, value))
                writer.add_scalar(key, value, epoch)
            else:
                writer.add_figure(key, (v), epoch)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(run_log_dir, 'epoch_%05d.pth' % epoch))

        eval_f1 = eval_results.get('eval/vuv_f1')
        eval_f0 = eval_results.get('eval/f0')
        improved = False

        if eval_f1 is not None:
            try:
                f1_value = float(eval_f1)
                if not math.isfinite(f1_value):
                    f1_value = None
            except (TypeError, ValueError):
                f1_value = None
        else:
            f1_value = None

        if eval_f0 is not None:
            try:
                f0_value = float(eval_f0)
                if not math.isfinite(f0_value):
                    f0_value = None
            except (TypeError, ValueError):
                f0_value = None
        else:
            f0_value = None

        if f1_value is not None:
            if f1_value > best_eval_f1 + metric_epsilon:
                improved = True
            elif (
                abs(f1_value - best_eval_f1) <= metric_epsilon
                and f0_value is not None
                and f0_value < best_eval_f0 - metric_epsilon
            ):
                improved = True

            if improved:
                best_eval_f1 = f1_value
                if f0_value is not None:
                    best_eval_f0 = f0_value
        else:
            eval_loss = eval_results.get('eval/loss')
            if eval_loss is not None and eval_loss < best_eval_loss:
                improved = True
                best_eval_loss = eval_loss

        if improved:
            if f1_value is not None:
                if f0_value is not None and math.isfinite(f0_value):
                    logger.info(
                        'New best model with eval/vuv_f1 %.4f (eval/f0 %.4f)',
                        best_eval_f1,
                        best_eval_f0,
                    )
                else:
                    logger.info('New best model with eval/vuv_f1 %.4f', best_eval_f1)
            else:
                logger.info('New best model with eval/loss %.4f', best_eval_loss)
            trainer.save_checkpoint(osp.join(run_log_dir, 'epoch_%05d.pth' % epoch))
            trainer.save_checkpoint(osp.join(run_log_dir, 'best.pth'), use_ema_model=True)

        if trainer.finish_train:
            logger.info(
                'Early stopping triggered at epoch %d with learning rate %.6e (threshold %.6e) after %d stagnant epochs on %s.',
                epoch,
                trainer._get_lr(),
                trainer.early_stop_min_lr,
                getattr(trainer, '_epochs_since_early_stop_improve', 0),
                trainer.early_stop_metric,
            )
            break

    writer.close()
    logger.removeHandler(file_handler)
    file_handler.close()


@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    config.setdefault('log_dir', 'Checkpoint')
    loss_params = copy.deepcopy(config.get('loss_params', {}))
    lambda_candidates = loss_params.pop('lambda_vuv_candidates', None)

    if lambda_candidates:
        base_log_dir = config.get('log_dir', 'Checkpoint')
        os.makedirs(base_log_dir, exist_ok=True)
        for candidate in lambda_candidates:
            candidate_value = float(candidate)
            candidate_config = copy.deepcopy(config)
            candidate_config['log_dir'] = osp.join(base_log_dir, f"lambda_vuv_{candidate_value:g}")
            candidate_config.setdefault('loss_params', {})['lambda_vuv'] = candidate_value
            candidate_config['loss_params'].pop('lambda_vuv_candidates', None)
            _train_single_run(candidate_config, config_path, candidate_config['log_dir'])
    else:
        config.setdefault('loss_params', {})
        config['loss_params']['lambda_vuv'] = float(config['loss_params'].get('lambda_vuv', 1.0))
        config['loss_params'].pop('lambda_vuv_candidates', None)
        _train_single_run(config, config_path, config['log_dir'])

    return 0

if __name__=="__main__":
    main()
