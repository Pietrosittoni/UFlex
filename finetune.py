"""Fine-tune a pretrained UFlex on a new dataset, then evaluate full rollouts.

The transformer backbone is initialized from ``pretrained_checkpoint``; the patch embedding
and output projection (which depend on the data's channels) are trained from scratch.
The backbone keeps a constant learning rate ``lr`` while the projections follow a cosine
schedule from ``projection_lr`` to ``projection_lr_min``.

    python finetune.py --config configs/cns_3d_finetune/uflex_b.yaml
"""

import argparse
import math

import torch
import wandb

from uflex.config import Config
from uflex.data import build_dataloaders
from uflex.engine import FieldTransform, errors_at_horizons, evaluate_rollout
from uflex.experiment import (
    PROJECTION_PREFIXES, build_model, build_optimizer, fit, init_wandb, load_backbone, save_checkpoint, set_seed,
)


def main(config: Config) -> None:
    set_seed(config.seed)
    init_wandb(config)

    train_loader, val_loader, rollout_loader = build_dataloaders(config)
    model = build_model(config, train_loader).to(config.device)
    load_backbone(model, config.pretrained_checkpoint, config.device)

    backbone = [p for n, p in model.named_parameters() if not n.startswith(PROJECTION_PREFIXES)]
    projection = [p for n, p in model.named_parameters() if n.startswith(PROJECTION_PREFIXES)]
    optimizer = build_optimizer(config, [
        {"params": backbone, "lr": float(config.lr)},
        {"params": projection, "lr": float(config.projection_lr)},
    ])

    total_steps = max(1, int(len(train_loader) * config.epochs / config.accumulation_steps))
    min_ratio = float(config.projection_lr_min) / float(config.projection_lr)

    def projection_schedule(step: int) -> float:
        progress = min(step, total_steps) / total_steps
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda step: 1.0, projection_schedule])

    model = fit(model, train_loader, val_loader, optimizer, scheduler, config)

    errors = evaluate_rollout(
        model, rollout_loader, FieldTransform(config.mean, config.std, config.log_scale, config.device),
        num_parameters=config.model.get("num_parameters", 0), device=config.device,
        error_space=config.get("rollout_error_space", "normalized"),
    )
    report = ", ".join(f"TE({100 * f:.0f}%) {e:.3f}" for f, e in errors_at_horizons(errors).items())
    print(f"Rollout test error over {len(errors)} steps: {report}")

    save_checkpoint(model, config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--device", help="Override the device, e.g. cuda:1.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.device:
        config.device = args.device
    main(config)
