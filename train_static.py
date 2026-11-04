"""Train UFlex on a steady-state task: map one state directly to another (no rollout).

Examples: the state of active_matter after a fixed horizon from its initial condition,
or the DiffusionPDE coefficient -> solution problems (darcy, helmholtz, ...).

    python train_steady_state.py --config configs/active_matter_steady/uflex_b.yaml
"""

import argparse

import wandb

from uflex.config import Config
from uflex.data import build_steady_state_dataloaders
from uflex.engine import FieldTransform, evaluate
from uflex.experiment import build_model, build_optimizer, build_scheduler, fit, init_wandb, save_checkpoint, set_seed


def main(config: Config) -> None:
    set_seed(config.seed)
    init_wandb(config)

    train_loader, val_loader = build_steady_state_dataloaders(config)
    model = build_model(config, train_loader).to(config.device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer, len(train_loader))

    # `fit` validates every `val_interval` epochs and finishes with the EMA weights.
    model = fit(model, train_loader, val_loader, optimizer, scheduler, config)

    metrics = evaluate(
        model, val_loader, FieldTransform(config.mean, config.std, config.log_scale, config.device),
        num_parameters=config.model.get("num_parameters", 0), device=config.device,
    )
    print(f"Final validation relative L2: {metrics['val/Relative_l2']:.4f}")

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
