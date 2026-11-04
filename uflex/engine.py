"""Training and evaluation loops: one-step training, one-step validation and autoregressive rollout."""

from typing import Dict, Optional, Sequence

import torch
import wandb
from torch import nn


def log(metrics: dict) -> None:
    """Log to Weights & Biases when a run is active, otherwise do nothing."""
    if wandb.run is not None:
        wandb.log(metrics)


class FieldTransform:
    """Maps physical fields to the space the model works in, and back.

    encode: per-channel standardization ``(x - mean) / std``, optionally followed by the
            symmetric log transform ``sign(x) * log(1 + |x|)`` for heavy-tailed fields.
    decode: the exact inverse.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float], log_scale: bool = False, device=None):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)  # broadcasts over the channel axis
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.log_scale = log_scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return torch.sign(x) * torch.log1p(torch.abs(x)) if self.log_scale else x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_scale:
            x = torch.sign(x) * torch.expm1(torch.abs(x))
        return x * self.std.to(x.device) + self.mean.to(x.device)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, per_frame: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Relative L2 error ``||prediction - target|| / ||target||``, averaged over the batch.

    With ``per_frame`` the error is computed separately for every frame (axis 1) and a
    vector of shape (T,) is returned; otherwise every sample is flattened into one vector.
    """
    keep = 2 if per_frame else 1
    diff = (prediction - target).flatten(keep).pow(2).sum(-1)
    norm = target.flatten(keep).pow(2).sum(-1) + eps
    return torch.sqrt(diff / norm).mean(dim=0)


def physical_parameters(batch: dict, num_parameters: int, device) -> torch.Tensor:
    """The conditioning parameters of a batch, or a dummy tensor of ones when the model is unconditioned."""
    if num_parameters > 0:
        return batch["constant_scalars"].to(device)
    return torch.ones(batch["input_fields"].shape[0], 1, device=device)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    transform: FieldTransform,
    num_parameters: int,
    device,
    mixed_precision: bool = True,
    scaler: Optional[torch.amp.GradScaler] = None,
    accumulation_steps: int = 1,
    ema=None,
    max_grad_norm: float = 1.0,
) -> None:
    """One epoch of one-step training with an MSE loss on the encoded fields."""
    model.train()
    loss_fn = nn.MSELoss()

    for step, batch in enumerate(loader):
        x = transform.encode(batch["input_fields"].to(device))
        y = transform.encode(batch["output_fields"].to(device))
        parameters = physical_parameters(batch, num_parameters, device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            loss = loss_fn(model(x, parameters=parameters), y)

        if scaler is not None:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        log({"learning rate": scheduler.get_last_lr()[0], "train/loss": loss.item(), "train/grad_norm": grad_norm})


@torch.no_grad()
def evaluate(model: nn.Module, loader, transform: FieldTransform, num_parameters: int, device) -> Dict[str, float]:
    """One-step validation: MSE and relative L2 error in encoded space."""
    model.eval()
    mse, rel_l2 = 0.0, 0.0
    for batch in loader:
        x = transform.encode(batch["input_fields"].to(device))
        y = transform.encode(batch["output_fields"].to(device))
        prediction = model(x, parameters=physical_parameters(batch, num_parameters, device))
        mse += nn.functional.mse_loss(prediction, y).item()
        rel_l2 += relative_l2(prediction, y).item()

    metrics = {"val/MSE_total": mse / len(loader), "val/Relative_l2": rel_l2 / len(loader)}
    log(metrics)
    return metrics


@torch.no_grad()
def rollout(model: nn.Module, x: torch.Tensor, parameters: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Autoregressive rollout.

    The model sees a sliding window of the last ``T_in`` frames; every prediction is appended
    to the window and the oldest frame dropped.

    Args:
        x: Initial frames (B, T_in, *spatial, C), in encoded space.
        num_steps: Number of frames to predict.

    Returns:
        Predicted frames (B, num_steps, *spatial, C) on the CPU.
    """
    window, predictions = x, []
    for _ in range(num_steps):
        next_frame = model(window, parameters=parameters)
        predictions.append(next_frame.cpu())
        window = torch.cat([window[:, 1:], next_frame], dim=1)
    return torch.cat(predictions, dim=1)


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    loader,
    transform: FieldTransform,
    num_parameters: int,
    device,
    error_space: str = "normalized",
) -> torch.Tensor:
    """Relative L2 error of full autoregressive rollouts on the test trajectories.

    Args:
        error_space: ``"normalized"`` measures the error on encoded fields, ``"physical"``
            on fields mapped back to physical units.

    Returns:
        Error per rollout step, shape (rollout_steps,).
    """
    if error_space not in ("normalized", "physical"):
        raise ValueError(f"error_space must be 'normalized' or 'physical', got {error_space!r}")
    model.eval()

    error_per_step = 0.0
    for batch in loader:
        x = transform.encode(batch["input_fields"].to(device))
        target = batch["output_fields"]
        predictions = rollout(model, x, physical_parameters(batch, num_parameters, device), target.shape[1])
        if error_space == "physical":
            predictions = transform.decode(predictions)
        else:
            target = transform.encode(target)
        error_per_step = error_per_step + relative_l2(predictions, target, per_frame=True)

    error_per_step = error_per_step / len(loader)
    log({"val_rollout/Relative_l2": error_per_step.mean().item()})
    if wandb.run is not None:
        table = wandb.Table(data=[[t, e] for t, e in enumerate(error_per_step.tolist())], columns=["Time", "Relative_l2"])
        wandb.log({"val_rollout/error_vs_time": wandb.plot.line(table, "Time", "Relative_l2", title="Relative_l2 vs Time (rollout)")})
    return error_per_step
