"""Experiment plumbing shared by the training scripts: seeding, model/optimizer construction, the epoch loop and checkpoints."""

import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from torch import nn

from uflex.config import Config
from uflex.ema import EMA
from uflex.engine import FieldTransform, evaluate, train_one_epoch
from uflex.model import UFlex

# Parameters of the patch embedding and output projection, i.e. everything that depends on the data's channels.
PROJECTION_PREFIXES = ("patch_norm", "lift_conv", "embed_norm", "proj")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_channels(loader) -> Tuple[int, int]:
    """Number of input and output fields, read from the first sample of a loader's dataset."""
    sample = loader.dataset[0]
    return sample["input_fields"].shape[-1], sample["output_fields"].shape[-1]


def build_model(config: Config, loader) -> UFlex:
    """Build UFlex from ``config.model``; the channel counts are read from the data."""
    in_channels, out_channels = infer_channels(loader)
    model = UFlex(in_channels=in_channels, out_channels=out_channels, **config.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"UFlex: {in_channels} -> {out_channels} fields, {num_params / 1e6:.1f}M parameters")
    log_to_wandb({"train/number_of_parameters": num_params})
    return model


def build_optimizer(config: Config, params) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params, lr=float(config.lr), weight_decay=float(config.weight_decay), betas=(config.beta1, config.beta2)
    )


def build_scheduler(config: Config, optimizer, steps_per_epoch: int):
    """Linear warm-up for ``warmup_steps`` optimizer steps, then cosine decay to ``learning_rate_min``."""
    total_steps = steps_per_epoch * config.epochs / config.accumulation_steps
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=config.warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - config.warmup_steps), eta_min=float(config.learning_rate_min)
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[config.warmup_steps])


def init_wandb(config: Config) -> None:
    wandb.init(
        project=config.get("wandb_project", "UFlex"),
        entity=config.get("wandb_entity"),
        name=config.get("run_name"),
        tags=config.get("tags"),
        config=config.to_dict(),
    )


def log_to_wandb(metrics: dict) -> None:
    if wandb.run is not None:
        wandb.log(metrics)


def fit(model: nn.Module, train_loader, val_loader, optimizer, scheduler, config: Config) -> nn.Module:
    """Train for ``config.epochs`` epochs, validating every ``val_interval`` epochs.

    Returns the trained model, holding the EMA weights when ``ema_decay`` is set.
    """
    device = config.device
    transform = FieldTransform(config.mean, config.std, config.log_scale, device)
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision else None
    ema = EMA(model, decay=config.ema_decay, device="cpu") if config.get("ema_decay") is not None else None
    trained = torch.compile(model) if config.get("compile", False) else model

    for epoch in range(config.epochs):
        log_to_wandb({"train/epoch": epoch})
        train_one_epoch(
            trained, train_loader, optimizer, scheduler, transform,
            num_parameters=config.model.get("num_parameters", 0), device=device,
            mixed_precision=config.mixed_precision, scaler=scaler,
            accumulation_steps=config.accumulation_steps, ema=ema,
        )
        if epoch % config.val_interval == 0:
            metrics = evaluate(trained, val_loader, transform, config.model.get("num_parameters", 0), device)
            print(f"epoch {epoch:4d} | val MSE {metrics['val/MSE_total']:.3e} | val rel. L2 {metrics['val/Relative_l2']:.4f}")

    if ema is not None:
        model.load_state_dict(ema.module.state_dict())
    return model


def run_name(config: Config) -> str:
    """Name of the current run: ``config.run_name`` if set, else the W&B run name, else a timestamp."""
    if config.get("run_name"):
        return config.run_name
    if wandb.run is not None and wandb.run.name:
        return wandb.run.name
    return time.strftime("run-%Y%m%d-%H%M%S")


def save_checkpoint(model: nn.Module, config: Config, name: Optional[str] = None) -> str:
    """Save ``model.pt`` (a plain state dict) and the resolved ``config.yaml`` under ``output_dir/name``."""
    directory = os.path.join(config.get("output_dir", "checkpoints"), name or run_name(config))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "model.pt")
    torch.save(model.state_dict(), path)
    config.save(os.path.join(directory, "config.yaml"))
    print(f"Saved checkpoint to {path}")
    return path


def load_backbone(model: nn.Module, checkpoint_path: str, device) -> None:
    """Load every weight of a checkpoint except the data-specific patch embedding and output projection."""
    state = torch.load(checkpoint_path, map_location=device)
    backbone = {k: v for k, v in state.items() if not k.startswith(PROJECTION_PREFIXES)}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    print(f"Loaded backbone from {checkpoint_path} ({len(missing)} projection tensors left at initialization)")
    if unexpected:
        print(f"Ignored unexpected keys: {unexpected}")
