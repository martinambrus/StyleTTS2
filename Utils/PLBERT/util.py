import os
import yaml
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def __init__(self, config):
        super().__init__(config)

        # Drop the unused pooler so DDP does not expect gradients for it.
        self.pooler = None
        if hasattr(self, "pooler_activation"):
            self.pooler_activation = nn.Identity()

    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"):
            ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
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
