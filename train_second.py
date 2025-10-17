from torch.utils.tensorboard import SummaryWriter
from meldataset import build_dataloader
from Utils.PLBERT.util import load_plbert
from models import build_model, load_ASR_models, load_checkpoint, load_F0_models
from utils import get_data_path_list, length_to_mask, log_norm, maximum_path, recursive_munch
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WhisperLoss
from Modules.slmadv import SLMAdversarialLoss, SkipSLMAdversarial
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer
from monotonic_align import mask_from_lens
from munch import Munch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger

import logging
import os
import yaml
import time
import numpy as np
import torch
import click
import shutil
import traceback
import torch.nn.functional as F
import random

from phoneme_dictionary import resolve_phoneme_dictionary_settings



def _clone_if_grad(tensor, *, force=False):
    if isinstance(tensor, torch.Tensor) and (force or tensor.requires_grad):
        return tensor.clone()
    return tensor


def _run_pitch_extractor(extractor, mel):
    outputs = extractor(mel)
    classifier = outputs
    detector = None
    features = None

    if isinstance(outputs, dict):
        classifier = (
            outputs.get('f0')
            or outputs.get('logits')
            or outputs.get('classification')
            or outputs.get('predictions')
        )
        detector = outputs.get('detector') or outputs.get('voicing')
        features = outputs.get('features')
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) >= 1:
            classifier = outputs[0]
        if len(outputs) >= 2:
            detector = outputs[1]
        if len(outputs) >= 3:
            features = outputs[2]

    def _standardize_pitch_tensor(tensor):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.float()
        if tensor.dim() >= 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        if tensor.dim() >= 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0, ...]
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()

    classifier = _standardize_pitch_tensor(classifier)
    detector = _standardize_pitch_tensor(detector)
    features = _standardize_pitch_tensor(features)

    if isinstance(classifier, torch.Tensor):
        classifier = torch.abs(classifier)

    return classifier, detector, features



def _run_text_aligner(aligner, mels, mask, texts):
    outputs = aligner(mels, mask, texts)
    if isinstance(outputs, dict):
        return (
            outputs.get('ctc_logits'),
            outputs.get('s2s_logits'),
            outputs.get('s2s_attn'),
        )
    return outputs


def _log_rank_debug(accelerator, message):
    try:
        rank = accelerator.process_index
    except Exception:
        rank = 'NA'
    print(f"[rank{rank}] {message}", flush=True)

logger = get_logger(__name__, log_level="DEBUG")


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Stage-two training swaps modules in and out of the graph (e.g., SLM adversarial
    # updates may skip batches entirely).  Keeping ``find_unused_parameters`` enabled
    # by default prevents DDP from stalling when a module's grads are legitimately
    # absent for an iteration.  Users can still override the behaviour from the YAML
    # config if they know their setup never skips parameters.
    find_unused = config.get('find_unused_parameters', True)
    broadcast_buffers = config.get('broadcast_buffers', False)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused,
        broadcast_buffers=broadcast_buffers,
    )
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])

    seed = config.get('seed', 42)
    set_seed(seed, device_specific=False)
    np.random.seed(seed)
    random.seed(seed)

    if accelerator.is_main_process:
        logger.info("Using global seed %d", seed)

    if accelerator.is_main_process:
        shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        writer = SummaryWriter(log_dir + "/tensorboard")

        file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.logger.addHandler(file_handler)
    else:
        writer = None

    accelerator.wait_for_everyone()


    batch_size = config.get('batch_size', 10)

    requested_epochs = config.get('epochs_2nd', 200)
    log_interval = config.get('log_interval', 10)
    save_frequency = config.get('save_freq', 2)

    data_params = config.get('data_params') or {}
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    asr_config_path = config.get('ASR_config') or None
    dictionary_source, dictionary_settings = resolve_phoneme_dictionary_settings(
        data_params=data_params,
        asr_config_path=asr_config_path,
    )

    dataset_config = {}
    if dictionary_source is not None:
        dataset_config['dict_path'] = dictionary_source
    if dictionary_settings:
        dataset_config['dictionary_config'] = dictionary_settings

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = accelerator.device

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=2,
        dataset_config=dataset_config,
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config=dataset_config,
    )

    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = asr_config_path
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(
            ASR_path,
            ASR_config,
            dictionary_path=dictionary_source,
            dictionary_config=dictionary_settings,
        )

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        F0_config = config.get('F0_config') or None
        pitch_extractor = load_F0_models(F0_path, F0_config)

        # load PL-BERT model
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

    accelerator.wait_for_everyone()

    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    steps_per_epoch = len(train_dataloader)

    unwrapped_models = {}
    for key in model:
        model[key] = accelerator.prepare(model[key])
        try:
            unwrapped_models[key] = accelerator.unwrap_model(model[key])
        except Exception:
            unwrapped_models[key] = model[key]

    predictor_module = unwrapped_models['predictor']

    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )

    def _disable_rng_sync(dataloader):
        if hasattr(dataloader, "rng_types"):
            dataloader.rng_types = []
            dataloader.synchronized_generator = None

    _disable_rng_sync(train_dataloader)
    _disable_rng_sync(val_dataloader)

    diffusion_module = unwrapped_models['diffusion']
    diffusion_impl = getattr(diffusion_module, 'diffusion', diffusion_module)
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    total_epochs = requested_epochs

    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = os.path.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            if accelerator.is_main_process:
                print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(
                model,
                None,
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion'],
            )

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            total_epochs = requested_epochs + start_epoch

            style_encoder_module = unwrapped_models['style_encoder']
            predictor_encoder_module = unwrapped_models['predictor_encoder']
            predictor_encoder_module.load_state_dict(style_encoder_module.state_dict())
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    slm_hop_length = getattr(
        model_params.slm,
        "hop_length",
        config["preprocess_params"]["spect_params"].get("hop_length", 300),
    )
    wl = WhisperLoss(
        model_params.slm.model,
        model.wd,
        sr,
        model_params.slm.sr,
        hop_length=slm_hop_length,
    ).to(device)
    
    sampler = DiffusionSampler(
        diffusion_impl,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": requested_epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    for key, opt in optimizer.optimizers.items():
        optimizer.optimizers[key] = accelerator.prepare(opt)
        optimizer.schedulers[key] = accelerator.prepare(optimizer.schedulers[key])
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    original_total_epochs = total_epochs

    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model,
            optimizer,
            config['pretrained_model'],
            load_only_params=config.get('load_only_params', True),
        )
        # advance start epoch or we'd re-train and rewrite the last epoch file
        start_epoch += 1
        accelerator.print('\nmodel data loaded, starting training epoch %05d\n' % start_epoch)

        if requested_epochs <= 0:
            total_epochs = start_epoch + 1
        elif requested_epochs <= start_epoch:
            total_epochs = start_epoch + requested_epochs
        else:
            total_epochs = max(total_epochs, requested_epochs)
    else:
        total_epochs = requested_epochs + start_epoch

    if accelerator.is_main_process and total_epochs > original_total_epochs:
        accelerator.print(
            f"Requested {requested_epochs} epochs but resuming from epoch {start_epoch}; extending target to {total_epochs}."
        )
    epochs_remaining = max(total_epochs - start_epoch, 0)
    accelerator.print(
        f"Stage-two training target: {total_epochs} epochs (starting from epoch {start_epoch}, {epochs_remaining} remaining)."
    )

    text_aligner_module = unwrapped_models['text_aligner']
    n_down = getattr(text_aligner_module, 'n_down')

    best_loss = float('inf')  # best test loss
    iters = 0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    accelerator.print('BERT', optimizer.optimizers['bert'])
    accelerator.print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(
        model,
        wl,
        sampler,
        slmadv_params.min_len,
        slmadv_params.max_len,
        batch_percentage=slmadv_params.batch_percentage,
        skip_update=slmadv_params.iter,
        sig=slmadv_params.sig,
        accelerator=accelerator,
        model_unwrapped=unwrapped_models,
    )


    for epoch in range(start_epoch, total_epochs):
        _log_rank_debug(accelerator, f"epoch {epoch}: entering pre-epoch barrier")
        accelerator.wait_for_everyone()
        _log_rank_debug(accelerator, f"epoch {epoch}: exited pre-epoch barrier")
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()


        if epoch >= diff_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                aligner_success = torch.tensor(1, device=device, dtype=torch.int)
                try:
                    _, _, s2s_attn = _run_text_aligner(model.text_aligner, mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except Exception:
                    aligner_success.zero_()

                if accelerator.num_processes > 1:
                    aligner_success = accelerator.gather(aligner_success)
                    all_aligner_success = bool(aligner_success.min().item())
                else:
                    all_aligner_success = bool(aligner_success.item())

                if not all_aligner_success:
                    continue

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze() # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            bert_dur = _clone_if_grad(bert_dur)
            bert_dur_sampler = _clone_if_grad(bert_dur.detach(), force=True)
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            
            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    diffusion_impl.sigma_data = s_trg.std(axis=-1).mean().item()  # batch-wise std estimation
                    running_std.append(diffusion_impl.sigma_data)

                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur_sampler,
                        embedding_scale=1,
                        features=_clone_if_grad(ref, force=True),
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1),
                        embedding=_clone_if_grad(bert_dur),
                        features=_clone_if_grad(ref),
                    ).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur_sampler,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = diffusion_impl(
                        s_trg.unsqueeze(1), embedding=_clone_if_grad(bert_dur)
                    ).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty = torch.zeros(1, device=device)
                loss_diff = torch.zeros(1, device=device)

            d, p = model.predictor(
                _clone_if_grad(d_en),
                _clone_if_grad(s_dur),
                                                    input_lengths,
                                                    s2s_attn_mono,
                                                    text_mask)
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            p_en = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            gt_valid = torch.tensor(1, device=device, dtype=torch.int)
            if gt.size(-1) < 80:
                gt_valid.zero_()

            if accelerator.num_processes > 1:
                gt_valid = accelerator.gather(gt_valid)
                all_gt_valid = bool(gt_valid.min().item())
            else:
                all_gt_valid = bool(gt_valid.item())

            if not all_gt_valid:
                continue

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            
            with torch.no_grad():
                F0_real, _, F0 = _run_pitch_extractor(model.pitch_extractor, gt.unsqueeze(1))
                if isinstance(F0, torch.Tensor):
                    F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(
                    _clone_if_grad(en), F0_real, N_real, _clone_if_grad(s)
                )

                if epoch >= joint_epoch:
                    # ground truth from recording
                    wav = y_rec_gt # use recording since decoder is tuned
                else:
                    # ground truth from reconstruction
                    wav = y_rec_gt_pred # use reconstruction since decoder is fixed

            F0_fake, N_fake = model.predictor(
                _clone_if_grad(p_en),
                _clone_if_grad(s_dur),
                forward_mode="f0",
            )

            y_rec = model.decoder(
                _clone_if_grad(en), F0_fake, N_fake, _clone_if_grad(s)
            )

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step('msd')
                optimizer.step('mpd')
            else:
                d_loss = torch.zeros(1, device=device)

            d_loss_value = accelerator.gather(d_loss.detach()).mean().item()

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            if start_ds:
                loss_gen_all = gl(wav, y_rec).mean()
            else:
                loss_gen_all = torch.zeros(1, device=device)
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = torch.zeros(1, device=device)
            loss_dur = torch.zeros(1, device=device)
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_F0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * loss_gen_all
                + loss_params.lambda_slm * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
            )

            loss_mel_value = accelerator.gather(loss_mel.detach()).mean().item()
            running_loss += loss_mel_value
            accelerator.backward(g_loss)

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')

            if epoch >= diff_epoch:
                optimizer.step('diffusion')

            if epoch >= joint_epoch:
                optimizer.step('style_encoder')
                optimizer.step('decoder')

                # randomly pick whether to use in-distribution text
                use_ind = bool(np.random.rand() < 0.5)

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                try:
                    slm_out = slmadv(
                        i,
                        y_rec_gt,
                        y_rec_gt_pred,
                        waves,
                        mel_input_length,
                        ref_texts,
                        ref_lengths,
                        use_ind,
                        s_trg.detach(),
                        ref if multispeaker else None,
                    )
                except SkipSLMAdversarial:
                    slm_out = None

                slm_available = torch.tensor(
                    0 if slm_out is None else 1,
                    device=device,
                    dtype=torch.int,
                )
                if accelerator.num_processes > 1:
                    slm_available = accelerator.gather(slm_available)
                    has_slm = bool(slm_available.min().item())
                else:
                    has_slm = bool(slm_available.item())

                if not has_slm:
                    slm_out = None

                if slm_out is None:
                    d_loss_slm = torch.zeros(1, device=device)
                    loss_gen_lm = torch.zeros(1, device=device)
                    loss_gen_lm_value = 0.0
                    d_loss_slm_value = 0.0
                    should_run_discriminator = False
                else:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    if not isinstance(d_loss_slm, torch.Tensor):
                        d_loss_slm = torch.tensor(d_loss_slm, device=device, dtype=loss_gen_lm.dtype)

                    loss_gen_lm_value = accelerator.gather(loss_gen_lm.detach()).mean().item()
                    d_loss_slm_value = accelerator.gather(d_loss_slm.detach()).mean().item()

                    disc_flag = torch.tensor(
                        1 if torch.any(d_loss_slm.detach() != 0) else 0,
                        device=device,
                        dtype=torch.int,
                    )
                    if accelerator.num_processes > 1:
                        disc_flag = accelerator.gather(disc_flag)
                        should_run_discriminator = bool(disc_flag.max().item())
                    else:
                        should_run_discriminator = bool(disc_flag.item())

                    if should_run_discriminator:
                        optimizer.zero_grad()
                        accelerator.backward(d_loss_slm)
                        optimizer.step('wd')
                    else:
                        optimizer.zero_grad()
                        accelerator.backward(loss_gen_lm)

                        total_norm = {}
                        for key in model.keys():
                            total_norm[key] = 0
                            parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                            for p in parameters:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm[key] += param_norm.item() ** 2
                            total_norm[key] = total_norm[key] ** 0.5

                        if total_norm.get('predictor', 0) > slmadv_params.thresh:
                            scale = 1 / max(total_norm['predictor'], 1e-12)
                            for key in model.keys():
                                for p in model[key].parameters():
                                    if p.grad is not None:
                                        p.grad *= scale

                        for p in predictor_module.duration_proj.parameters():
                            if p.grad is not None:
                                p.grad *= slmadv_params.scale

                        for p in predictor_module.lstm.parameters():
                            if p.grad is not None:
                                p.grad *= slmadv_params.scale

                        for p in model.diffusion.parameters():
                            if p.grad is not None:
                                p.grad *= slmadv_params.scale

                        optimizer.step('bert_encoder')
                        optimizer.step('bert')
                        optimizer.step('predictor')
                        optimizer.step('diffusion')
                if slm_out is None:
                    should_run_discriminator = False

            else:
                d_loss_slm = torch.zeros(1, device=device)
                loss_gen_lm = torch.zeros(1, device=device)
                loss_gen_lm_value = 0.0
                d_loss_slm_value = 0.0

            loss_gen_all_value = accelerator.gather(loss_gen_all.detach()).mean().item()
            loss_lm_value = accelerator.gather(loss_lm.detach()).mean().item()
            loss_ce_value = accelerator.gather(loss_ce.detach()).mean().item()
            loss_dur_value = accelerator.gather(loss_dur.detach()).mean().item()
            loss_norm_rec_value = accelerator.gather(loss_norm_rec.detach()).mean().item()
            loss_F0_rec_value = accelerator.gather(loss_F0_rec.detach()).mean().item()
            loss_sty_value = accelerator.gather(loss_sty.detach()).mean().item()
            loss_diff_value = accelerator.gather(loss_diff.detach()).mean().item()
                
            iters = iters + 1

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                logger.info(
                    'Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                    % (
                        epoch + 1,
                        total_epochs,
                        i + 1,
                        len(train_list) // batch_size,
                        running_loss / log_interval,
                        d_loss_value,
                        loss_dur_value,
                        loss_ce_value,
                        loss_norm_rec_value,
                        loss_F0_rec_value,
                        loss_lm_value,
                        loss_gen_all_value,
                        loss_sty_value,
                        loss_diff_value,
                        d_loss_slm_value,
                        loss_gen_lm_value,
                    )
                )

                if writer is not None:
                    writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                    writer.add_scalar('train/gen_loss', loss_gen_all_value, iters)
                    writer.add_scalar('train/d_loss', d_loss_value, iters)
                    writer.add_scalar('train/ce_loss', loss_ce_value, iters)
                    writer.add_scalar('train/dur_loss', loss_dur_value, iters)
                    writer.add_scalar('train/slm_loss', loss_lm_value, iters)
                    writer.add_scalar('train/norm_loss', loss_norm_rec_value, iters)
                    writer.add_scalar('train/F0_loss', loss_F0_rec_value, iters)
                    writer.add_scalar('train/sty_loss', loss_sty_value, iters)
                    writer.add_scalar('train/diff_loss', loss_diff_value, iters)
                    writer.add_scalar('train/d_loss_slm', d_loss_slm_value, iters)
                    writer.add_scalar('train/gen_loss_slm', loss_gen_lm_value, iters)

                running_loss = 0

                accelerator.print('Time elasped:', time.time() - start_time)
                
        loss_test = 0.0
        loss_align = 0.0
        loss_f = 0.0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = _run_text_aligner(model.text_aligner, mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor(
                        p_en,
                        _clone_if_grad(s),
                        forward_mode="f0",
                    )

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(
                        _clone_if_grad(en), F0_fake, N_fake, _clone_if_grad(s)
                    )
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = _run_pitch_extractor(model.pitch_extractor, gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += accelerator.gather(loss_mel.detach()).mean().item()
                    loss_align += accelerator.gather(loss_dur.detach()).mean().item()
                    loss_f += accelerator.gather(loss_F0.detach()).mean().item()

                    iters_test += 1
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"Encountered exception: {e}")
                        traceback.print_exc()
                    continue

        if accelerator.is_main_process:
            accelerator.print('Epochs:', epoch + 1)
            logger.info(
                'Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (
                    loss_test / max(iters_test, 1),
                    loss_align / max(iters_test, 1),
                    loss_f / max(iters_test, 1),
                )
                + '\n\n\n'
            )
            accelerator.print('\n\n\n')
            if writer is not None:
                writer.add_scalar('eval/mel_loss', loss_test / max(iters_test, 1), epoch + 1)
                writer.add_scalar('eval/dur_loss', loss_align / max(iters_test, 1), epoch + 1)
                writer.add_scalar('eval/F0_loss', loss_f / max(iters_test, 1), epoch + 1)
        
        if accelerator.is_main_process and writer is not None:
            if epoch < joint_epoch:
                # generating reconstruction examples with GT duration

                with torch.no_grad():
                    for bib in range(len(asr)):
                        mel_length = int(mel_input_length[bib].item())
                        gt = mels[bib, :, :mel_length].unsqueeze(0)
                        en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                        F0_real, _, _ = _run_pitch_extractor(model.pitch_extractor, gt.unsqueeze(1))
                        s = model.style_encoder(gt.unsqueeze(1))
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                        y_rec = model.decoder(
                            _clone_if_grad(en), F0_real, real_norm, _clone_if_grad(s)
                        )

                        writer.add_audio('eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                        s_dur = model.predictor_encoder(gt.unsqueeze(1))
                        p_en = p[bib, :, :mel_length // 2].unsqueeze(0)

                        F0_fake, N_fake = model.predictor(
                            p_en,
                            _clone_if_grad(s_dur),
                            forward_mode="f0",
                        )

                        y_pred = model.decoder(
                            _clone_if_grad(en), F0_fake, N_fake, _clone_if_grad(s)
                        )

                        writer.add_audio('pred/y' + str(bib), y_pred.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                        if epoch == 0:
                            writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                        if bib >= 5:
                            break
            else:
                # generating sampled speech from text directly
                with torch.no_grad():
                    # compute reference styles
                    if multispeaker and epoch >= diff_epoch:
                        ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                        ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref_s = torch.cat([ref_ss, ref_sp], dim=1)

                    for bib in range(len(d_en)):
                        if multispeaker:
                            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                              embedding=bert_dur[bib].unsqueeze(0),
                              embedding_scale=1,
                                features=ref_s[bib].unsqueeze(0), # reference from the same speaker as the embedding
                                 num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device), 
                              embedding=bert_dur[bib].unsqueeze(0),
                              embedding_scale=1,
                                 num_steps=5).squeeze(1)

                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]

                    d = predictor_module.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0),
                                                     s, input_lengths[bib, ...].unsqueeze(0), text_mask[bib, :input_lengths[bib]].unsqueeze(0))

                    x, _ = predictor_module.lstm(d)
                    duration = predictor_module.duration_proj(x)

                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = (
                        d.transpose(-1, -2)
                        @ pred_aln_trg.unsqueeze(0).to(texts.device)
                    )
                    F0_pred, N_pred = model.predictor(
                        _clone_if_grad(en),
                        _clone_if_grad(s),
                        forward_mode="f0",
                    )
                    decoder_input = (
                        t_en[bib, :, :input_lengths[bib]]
                        .unsqueeze(0)
                        @ pred_aln_trg.unsqueeze(0).to(texts.device)
                    )
                    out = model.decoder(
                        _clone_if_grad(decoder_input),
                        F0_pred,
                        N_pred,
                        _clone_if_grad(ref.squeeze().unsqueeze(0)),
                    )

                    writer.add_audio('pred/y' + str(bib), out.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
                            
        _log_rank_debug(accelerator, f"epoch {epoch}: entering post-epoch barrier before save check")
        accelerator.wait_for_everyone()
        _log_rank_debug(accelerator, f"epoch {epoch}: exited post-epoch barrier before save check")
        save_this_epoch = (epoch % save_frequency == 0)
        if save_this_epoch:
            _log_rank_debug(
                accelerator,
                f"epoch {epoch}: synchronizing before checkpoint save",
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if (loss_test / max(iters_test, 1)) < best_loss:
                    best_loss = loss_test / max(iters_test, 1)
                accelerator.print('Saving..')
                state = {
                    'net':  {key: accelerator.unwrap_model(model[key]).state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test / max(iters_test, 1),
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
                _log_rank_debug(
                    accelerator,
                    f"epoch {epoch}: main process saving checkpoint to {save_path}",
                )
                accelerator.save(state, save_path)

                # if estimate sigma, save the estimated simga
                if model_params.diffusion.dist.estimate_sigma_data:
                    config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                    with open(
                        os.path.join(log_dir, os.path.basename(config_path)), 'w'
                    ) as outfile:
                        yaml.dump(config, outfile, default_flow_style=True)
            accelerator.wait_for_everyone()

    _log_rank_debug(accelerator, "final checkpoint: synchronizing before save")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print('Saving..')
        state = {
            'net':  {key: accelerator.unwrap_model(model[key]).state_dict() for key in model},
            'optimizer': optimizer.state_dict(),
            'iters': iters,
            'val_loss': loss_test / max(iters_test, 1),
            'epoch': epoch,
        }
        save_path = os.path.join(log_dir, config.get('second_stage_path', 'second_stage.pth'))
        _log_rank_debug(accelerator, f"final checkpoint path on main process: {save_path}")
        accelerator.save(state, save_path)
    accelerator.wait_for_everyone()
    _log_rank_debug(accelerator, "final checkpoint save section completed")
    accelerator.end_training()

if __name__=="__main__":
    main()
