import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


def _clone_if_grad(tensor, *, force=False):
    if isinstance(tensor, torch.Tensor) and (force or tensor.requires_grad):
        return tensor.clone()
    return tensor


class SkipSLMAdversarial(Exception):
    pass

class SLMAdversarialLoss(torch.nn.Module):

    def __init__(
        self,
        model,
        wl,
        sampler,
        min_len,
        max_len,
        batch_percentage=0.5,
        skip_update=10,
        sig=1.5,
        accelerator=None,
        model_unwrapped=None,
    ):
        super(SLMAdversarialLoss, self).__init__()
        self.model = model
        self.wl = wl
        self.sampler = sampler

        self.min_len = min_len
        self.max_len = max_len
        self.batch_percentage = batch_percentage

        self.sig = sig
        self.skip_update = skip_update
        self.accelerator = accelerator
        self.model_unwrapped = model_unwrapped or {}

    def _module(self, key):
        if isinstance(self.model_unwrapped, dict) and key in self.model_unwrapped:
            return self.model_unwrapped[key]
        module = self.model[key] if isinstance(self.model, dict) else getattr(self.model, key)
        if hasattr(module, "module"):
            return module.module
        return module

    def _shared_bool(self, proba: float, device: torch.device) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return np.random.rand() < proba

        if dist.get_rank() == 0:
            value = torch.rand(1, device=device) < proba
            flag = value.to(torch.bool)
        else:
            flag = torch.zeros(1, dtype=torch.bool, device=device)

        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def _shared_randint(self, low: int, high: int, device: torch.device) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return int(np.random.randint(low, high))

        if dist.get_rank() == 0:
            value = torch.randint(low, high, (1,), device=device)
        else:
            value = torch.zeros(1, dtype=torch.long, device=device)

        dist.broadcast(value, src=0)
        return int(value.item())

    def _maybe_resize_sampler_embedding(self, embedding: torch.Tensor) -> None:
        sampler = getattr(self, "sampler", None)
        if sampler is None:
            return

        denoise_fn = getattr(sampler, "denoise_fn", None)
        owner = getattr(denoise_fn, "__self__", None)
        net = getattr(owner, "net", None)
        if net is None:
            return

        net = getattr(net, "module", net)
        fixed_embedding = getattr(net, "fixed_embedding", None)
        if fixed_embedding is None:
            return

        length = embedding.size(1)
        device = embedding.device

        if fixed_embedding.embedding.weight.device != device:
            fixed_embedding.embedding = fixed_embedding.embedding.to(device)

        fixed_embedding._resize_if_needed(length, device)

    def forward(self, iters, y_rec_gt, y_rec_gt_pred, waves, mel_input_length, ref_text, ref_lengths, use_ind, s_trg, ref_s=None):
        text_mask = length_to_mask(ref_lengths).to(ref_text.device)
        bert_dur = self.model.bert(ref_text, attention_mask=(~text_mask).int())
        bert_dur = _clone_if_grad(bert_dur)
        bert_dur_sampler = _clone_if_grad(bert_dur.detach(), force=True)
        self._maybe_resize_sampler_embedding(bert_dur_sampler)
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        device = ref_text.device

        if dist.is_available() and dist.is_initialized():
            use_ind_tensor = torch.tensor(
                [1 if use_ind else 0],
                device=device,
                dtype=torch.long,
            )
            dist.broadcast(use_ind_tensor, src=0)
            use_ind = bool(use_ind_tensor.item())

        skip_sampler = False
        if use_ind:
            skip_sampler = self._shared_bool(0.5, device)

        if skip_sampler:
            s_preds = s_trg
        else:
            num_steps = self._shared_randint(3, 5, device)
            if ref_s is not None:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur_sampler,
                    embedding_scale=1,
                    features=_clone_if_grad(ref_s, force=True),
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
            else:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur_sampler,
                    embedding_scale=1,
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
            
        s_dur = s_preds[:, 128:]
        s_preds[:, :128]
        
        d, _ = self.model.predictor(
            _clone_if_grad(d_en),
            _clone_if_grad(s_dur),
            ref_lengths,
            torch.randn(ref_lengths.shape[0], ref_lengths.max(), 2).to(ref_text.device),
            text_mask,
        )
        
        bib = 0

        output_lengths = []
        attn_preds = []
        
        # differentiable duration modeling
        for _s2s_pred, _text_length in zip(d, ref_lengths):

            _s2s_pred_org = _s2s_pred[:_text_length, :]

            _s2s_pred = torch.sigmoid(_s2s_pred_org)
            _dur_pred = _s2s_pred.sum(axis=-1)

            length = int(torch.round(_s2s_pred.sum()).item())
            t = torch.arange(0, length).expand(length)

            t = torch.arange(0, length).unsqueeze(0).expand((len(_s2s_pred), length)).to(ref_text.device)
            loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2

            h = torch.exp(-0.5 * torch.square(t - (length - loc.unsqueeze(-1))) / (self.sig)**2)

            out = torch.nn.functional.conv1d(_s2s_pred_org.unsqueeze(0), 
                                         h.unsqueeze(1), 
                                         padding=h.shape[-1] - 1, groups=int(_text_length))[..., :length]
            attn_preds.append(F.softmax(out.squeeze(), dim=0))

            output_lengths.append(length)

        max_len = max(output_lengths)
        
        with torch.no_grad():
            t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)
            
        s2s_attn = torch.zeros(len(ref_lengths), int(ref_lengths.max()), max_len).to(ref_text.device)
        for bib in range(len(output_lengths)):
            s2s_attn[bib, :ref_lengths[bib], :output_lengths[bib]] = attn_preds[bib]

        asr_pred = t_en @ s2s_attn

        _, p_pred = self.model.predictor(
            _clone_if_grad(d_en),
            _clone_if_grad(s_dur),
            ref_lengths,
            s2s_attn,
            text_mask,
        )
        
        mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
        mel_len = min(mel_len, self.max_len // 2)
        
        # get clips
        en = []
        p_en = []
        sp = []
        wav = []

        for bib in range(len(output_lengths)):
            mel_length_pred = output_lengths[bib]
            mel_length_gt = int(mel_input_length[bib].item() / 2)
            if mel_length_gt <= mel_len or mel_length_pred <= mel_len:
                continue

            sp.append(s_preds[bib])

            random_start = np.random.randint(0, mel_length_pred - mel_len)
            en.append(asr_pred[bib, :, random_start:random_start+mel_len])
            p_en.append(p_pred[bib, :, random_start:random_start+mel_len])

            # get ground truth clips
            random_start = np.random.randint(0, mel_length_gt - mel_len)
            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y).to(ref_text.device))
            
            if len(wav) >= self.batch_percentage * len(waves): # prevent OOM due to longer lengths
                break

        batch_size_tensor = torch.tensor([len(sp)], device=ref_text.device)
        if self.accelerator is not None:
            global_min_batch = self.accelerator.gather(batch_size_tensor).min().item()
        else:
            global_min_batch = batch_size_tensor.min().item()

        if global_min_batch <= 1:
            raise SkipSLMAdversarial("skip slmadv")
            
        sp = torch.stack(sp)
        wav = torch.stack(wav).float()
        en = torch.stack(en)
        p_en = torch.stack(p_en)
        
        predictor_ddp = self.model.predictor
        predictor_module = self._module('predictor')
        decoder_module = self._module('decoder')

        prosody_style = _clone_if_grad(sp[:, 128:])
        acoustic_style = _clone_if_grad(sp[:, :128])

        F0_fake, N_fake = predictor_ddp(
            _clone_if_grad(p_en), prosody_style, forward_mode="f0"
        )
        y_pred = decoder_module(
            _clone_if_grad(en), F0_fake, N_fake, acoustic_style
        )
        
        # discriminator loss
        if (iters + 1) % self.skip_update == 0:
            if np.random.randint(0, 2) == 0:
                wav = y_rec_gt_pred
                use_rec = True
            else:
                use_rec = False

            crop_size = min(wav.size(-1), y_pred.size(-1))
            if use_rec: # use reconstructed (shorter lengths), do length invariant regularization
                if wav.size(-1) > y_pred.size(-1):
                    real_GP = wav[:, : , :crop_size]
                    out_crop = self.wl.discriminator_forward(real_GP.detach().squeeze())
                    out_org = self.wl.discriminator_forward(wav.detach().squeeze())
                    loss_reg = F.l1_loss(out_crop, out_org[..., :out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(real_GP.detach().squeeze(), y_pred.detach().squeeze()).mean()
                    else:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
                else:
                    real_GP = y_pred[:, : , :crop_size]
                    out_crop = self.wl.discriminator_forward(real_GP.detach().squeeze())
                    out_org = self.wl.discriminator_forward(y_pred.detach().squeeze())
                    loss_reg = F.l1_loss(out_crop, out_org[..., :out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), real_GP.detach().squeeze()).mean()
                    else:
                        d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
                
                # regularization (ignore length variation)
                d_loss += loss_reg

                out_gt = self.wl.discriminator_forward(y_rec_gt.detach().squeeze())
                out_rec = self.wl.discriminator_forward(y_rec_gt_pred.detach().squeeze())

                # regularization (ignore reconstruction artifacts)
                d_loss += F.l1_loss(out_gt, out_rec)

            else:
                d_loss = self.wl.discriminator(wav.detach().squeeze(), y_pred.detach().squeeze()).mean()
        else:
            d_loss = 0
            
        # generator loss
        gen_loss = self.wl.generator(y_pred.squeeze())
        
        gen_loss = gen_loss.mean()
        
        return d_loss, gen_loss, y_pred.detach().cpu().numpy()
    
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask
