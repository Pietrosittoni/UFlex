"""Train UFlex as an autoregressive one-step predictor, then evaluate full rollouts.

    python train.py --config configs/active_matter/uflex_b.yaml
"""

import argparse

import wandb

from uflex.config import Config
from uflex.data import build_dataloaders
from uflex.engine import FieldTransform, errors_at_horizons, evaluate_rollout
from uflex.experiment import build_model, build_optimizer, build_scheduler, fit, init_wandb, save_checkpoint, set_seed


def main(config: Config) -> None:
    set_seed(config.seed)
    init_wandb(config)

    train_loader, val_loader, rollout_loader = build_dataloaders(config)
    model = build_model(config, train_loader).to(config.device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer, len(train_loader))

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
