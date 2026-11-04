"""Tests for the training utilities and the released configs."""

import glob
import os

import pytest
import torch

from uflex import UFlex
from uflex.config import Config
from uflex.engine import FieldTransform, errors_at_horizons, relative_l2, rollout

CONFIGS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*", "*.yaml")))


@pytest.mark.parametrize("log_scale", [False, True])
def test_field_transform_round_trip(log_scale):
    transform = FieldTransform(mean=[1.0, -2.0], std=[0.5, 3.0], log_scale=log_scale)
    x = 10 * torch.randn(2, 1, 8, 8, 2)
    torch.testing.assert_close(transform.decode(transform.encode(x)), x, rtol=1e-5, atol=1e-4)


def test_relative_l2():
    target = torch.ones(2, 3, 4, 4, 1)
    prediction = target.clone()
    prediction[:, 1] *= 1.5
    assert relative_l2(target, target).item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(relative_l2(prediction, target, per_frame=True), torch.tensor([0.0, 0.5, 0.0]))


def test_errors_at_horizons():
    errors = torch.arange(1.0, 11.0)  # steps 1..10
    te = errors_at_horizons(errors, fractions=(0.3, 1.0))
    assert te[0.3] == pytest.approx(2.0)   # mean of the first 3 steps
    assert te[1.0] == pytest.approx(5.5)   # mean of all 10 steps


def test_rollout_uses_a_sliding_window():
    class Recorder(torch.nn.Module):
        """Predicts the sum of the window and records the windows it was given."""

        def __init__(self):
            super().__init__()
            self.windows = []

        def forward(self, x, parameters):
            self.windows.append(x[:, :, 0, 0, 0].clone())
            return x.sum(dim=1, keepdim=True)

    model = Recorder()
    x = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
    predictions = rollout(model, x, torch.ones(1, 1), num_steps=3)
    assert predictions.flatten().tolist() == [3.0, 5.0, 8.0]
    assert [w.flatten().tolist() for w in model.windows] == [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.relpath(p, os.path.dirname(CONFIGS[0]) + "/.."))
def test_released_configs_build_and_run(path):
    """Every config builds a model that runs on a minimal input of its dimensionality."""
    config = Config.from_yaml(path)
    for key in ["dataset_name", "data_path", "input_steps", "mean", "std", "epochs", "batch_size", "lr"]:
        assert hasattr(config, key), f"{path} misses {key}"
    channels = len(config.mean)
    model = UFlex(in_channels=channels, **config.model).eval()

    # Smallest valid input: one level-merge group per frame along the first axis, one patch along the others.
    patch = list(config.model["patch_size"])
    spatial = [patch[0] * config.model["downsampling_factor"] ** config.model["depth"]] + patch[1:]
    frames = config.input_steps
    num_parameters = max(config.model.get("num_parameters", 0), 1)
    with torch.no_grad():
        y = model(torch.randn(1, frames, *spatial, channels), torch.ones(1, num_parameters))
    assert y.shape == (1, 1, *spatial, channels)
