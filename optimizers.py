#coding:utf-8
import torch
from functools import reduce
from torch.optim import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
                self._resize_state_tensors(self.optimizers[key])
            except Exception:
                print("Unloaded %s" % key)

    def _unwrap_optimizer(self, optimizer):
        base_optimizer = optimizer
        visited = set()

        while hasattr(base_optimizer, "optimizer"):
            next_optimizer = base_optimizer.optimizer
            if next_optimizer is None or next_optimizer in visited:
                break
            visited.add(next_optimizer)
            base_optimizer = next_optimizer

        return base_optimizer

    def _resize_state_tensors(self, optimizer):
        base_optimizer = self._unwrap_optimizer(optimizer)

        if not hasattr(base_optimizer, "state"):
            return

        state = base_optimizer.state

        for param, param_state in state.items():
            if param is None or not isinstance(param_state, dict):
                continue

            param_shape = tuple(param.shape)

            for name, value in list(param_state.items()):
                if not torch.is_tensor(value):
                    continue

                if tuple(value.shape) == param_shape:
                    continue

                new_tensor = torch.zeros_like(param.data)

                value_converted = value.to(new_tensor.device, new_tensor.dtype)

                if value_converted.dim() != new_tensor.dim():
                    param_state[name] = new_tensor
                    continue

                slices = [
                    slice(0, min(old_dim, new_dim))
                    for old_dim, new_dim in zip(value_converted.shape, new_tensor.shape)
                ]

                if slices:
                    new_tensor[tuple(slices)] = value_converted[tuple(slices)]
                else:
                    new_tensor.copy_(value_converted)

                param_state[name] = new_tensor

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None, set_to_none=True):
        if key is not None:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)
        else:
            _ = [self.optimizers[key].zero_grad(set_to_none=set_to_none) for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 2e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        div_factor=1,
        final_div_factor=1)

    return scheduler

def build_optimizer(parameters_dict, scheduler_params_dict, lr):
    optim = dict([(key, AdamW(params, lr=lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9))
                   for key, params in parameters_dict.items()])

    schedulers = dict([(key, define_scheduler(opt, scheduler_params_dict[key])) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim
