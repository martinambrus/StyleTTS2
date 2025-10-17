import os
import yaml
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel


def _runtime_max_positions(config, override=None):
    """Return the runtime position capacity for PL-BERT."""

    if override is None:
        override = 0

    env_override = int(os.environ.get("PLBERT_MAX_POSITION", 0))
    override = max(int(override or 0), env_override)

    if override <= 0:
        override = 1024

    return max(int(config.max_position_embeddings), override)

class CustomAlbert(AlbertModel):
    def __init__(self, config):
        super().__init__(config)

        # Drop the unused pooler so DDP does not expect gradients for it.
        self.pooler = None
        if hasattr(self, "pooler_activation"):
            self.pooler_activation = nn.Identity()

    def resize_position_embeddings(self, target_positions):
        """Resize the learned position embeddings to ``target_positions``."""

        current_size, embed_dim = self.embeddings.position_embeddings.weight.shape

        if target_positions == current_size:
            return self.embeddings.position_embeddings

        if target_positions <= 0:
            raise ValueError("target_positions must be positive")

        old_weight = self.embeddings.position_embeddings.weight.data
        device = old_weight.device
        dtype = old_weight.dtype

        new_embeddings = nn.Embedding(target_positions, embed_dim)
        new_weight = new_embeddings.weight.data

        copy_size = min(current_size, target_positions)
        new_weight[:copy_size].copy_(old_weight[:copy_size])

        if target_positions > current_size:
            pad_count = target_positions - current_size
            new_weight[current_size:].copy_(old_weight[-1:].expand(pad_count, -1))

        new_embeddings = new_embeddings.to(device=device, dtype=dtype)
        self.embeddings.position_embeddings = new_embeddings

        position_ids = torch.arange(target_positions, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        self.embeddings.register_buffer("position_ids", position_ids, persistent=False)

        self.config.max_position_embeddings = target_positions

        return self.embeddings.position_embeddings

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        if isinstance(input_ids, torch.Tensor):
            kwargs.setdefault("token_type_ids", torch.zeros_like(input_ids))

        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    model_params = dict(plbert_config['model_params'])
    albert_base_configuration = AlbertConfig(**model_params)
    bert = CustomAlbert(albert_base_configuration)

    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"):
            ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(
        log_dir + "/step_" + str(iters) + ".t7",
        map_location='cpu',
        weights_only=False,
    )
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    new_state_dict.pop("embeddings.position_ids", None)
    bert.load_state_dict(new_state_dict, strict=False)

    target_positions = _runtime_max_positions(
        albert_base_configuration,
        override=model_params.get("runtime_max_position_embeddings"),
    )

    if target_positions > bert.config.max_position_embeddings:
        bert.resize_position_embeddings(target_positions)
        embeddings = bert.embeddings
        if hasattr(embeddings, "token_type_ids"):
            token_type_ids = torch.zeros(
                (1, target_positions),
                dtype=embeddings.token_type_ids.dtype,
                device=embeddings.token_type_ids.device,
            )
            embeddings.register_buffer("token_type_ids", token_type_ids, persistent=False)

    # The pooler weights are unused in downstream training loops but remain
    # trainable parameters in the default ALBERT implementation. Under
    # distributed training, parameters that never contribute to the loss cause
    # DDP to error out unless ``find_unused_parameters`` is enabled. Replacing
    # the pooler with an identity module removes those stale parameters and
    # keeps the distributed graph consistent without the performance penalty of
    # enabling unused-parameter detection.
    if hasattr(bert, "pooler"):
        bert.pooler = None
    if hasattr(bert, "pooler_activation"):
        bert.pooler_activation = nn.Identity()

    return bert
