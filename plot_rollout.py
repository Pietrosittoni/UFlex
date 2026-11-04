"""Plot an autoregressive rollout of a trained UFlex against the reference trajectory.

    python plot_rollout.py --config configs/planetswe/uflex.yaml \
        --checkpoint checkpoints/<run>/model.pt --out rollout.png

For 3D data the middle slice along the second spatial axis is shown.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from uflex.config import Config
from uflex.data import build_dataloaders
from uflex.engine import FieldTransform, physical_parameters, rollout
from uflex.experiment import build_model


def plot_trajectories(prediction: np.ndarray, reference: np.ndarray, stride: int, path: str, num_frames: int = 10) -> None:
    """Two rows of snapshots (prediction on top, reference below) at evenly spaced times; arrays are (T, H, W)."""
    length = min(len(prediction), len(reference))
    times = np.linspace(0, length - 1, min(num_frames, length), dtype=int)
    vmin, vmax = min(prediction.min(), reference.min()), max(prediction.max(), reference.max())

    fig, axes = plt.subplots(2, len(times), figsize=(2.1 * len(times), 4.4), squeeze=False)
    for col, t in enumerate(times):
        for row, (field, label) in enumerate([(prediction, "Prediction"), (reference, "Reference")]):
            ax = axes[row, col]
            ax.imshow(field[t], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
        axes[1, col].set_xlabel(f"t={(t + 1) * stride}", fontsize=11)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main(args) -> None:
    config = Config.from_yaml(args.config)
    device = args.device or config.device

    _, _, rollout_loader = build_dataloaders(config)
    model = build_model(config, rollout_loader).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    transform = FieldTransform(config.mean, config.std, config.log_scale, device)
    batch = next(iter(rollout_loader))
    x = transform.encode(batch["input_fields"].to(device))
    target = batch["output_fields"]
    parameters = physical_parameters(batch, config.model.get("num_parameters", 0), device)
    prediction = transform.decode(rollout(model, x, parameters, target.shape[1]))

    sample = args.sample % prediction.shape[0]
    prediction, target = prediction[sample, ..., args.channel], target[sample, ..., args.channel]
    if prediction.ndim == 4:  # 3D: (T, H, W, D) -> middle slice along W
        middle = prediction.shape[2] // 2
        prediction, target = prediction[:, :, middle], target[:, :, middle]

    plot_trajectories(prediction.numpy(), target.numpy(), config.stride, args.out, args.frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Config the checkpoint was trained with.")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt.")
    parser.add_argument("--out", default="rollout.png", help="Output image.")
    parser.add_argument("--sample", type=int, default=0, help="Trajectory index within the first test batch.")
    parser.add_argument("--channel", type=int, default=0, help="Field (channel) to plot.")
    parser.add_argument("--frames", type=int, default=10, help="Number of snapshots.")
    parser.add_argument("--device", help="Override the device, e.g. cuda:1.")
    main(parser.parse_args())
