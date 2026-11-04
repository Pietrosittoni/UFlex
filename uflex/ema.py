"""Exponential moving average (EMA) of model weights."""

from copy import deepcopy

import torch
from torch import nn


class EMA(nn.Module):
    """Keeps ``ema = decay * ema + (1 - decay) * weights`` for every parameter and buffer.

    Args:
        model: The model to track (it is deep-copied).
        decay: EMA decay per update.
        device: Device holding the averaged copy, e.g. ``"cpu"`` to save GPU memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def _update(self, model: nn.Module, update_fn) -> None:
        for ema_value, model_value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_value = model_value.to(device=self.device)
            ema_value.copy_(update_fn(ema_value, model_value))

    def update(self, model: nn.Module) -> None:
        self._update(model, lambda ema, current: self.decay * ema + (1.0 - self.decay) * current)

    def set(self, model: nn.Module) -> None:
        self._update(model, lambda ema, current: current)
