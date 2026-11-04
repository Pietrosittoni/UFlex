"""Static prediction task: predict the state ``horizon`` steps ahead directly from the initial condition."""

import torch
from the_well.data import WellDataset
from torch.utils.data import Dataset


class StaticPredictionDataset(Dataset):
    """Pairs (u_0, u_horizon) from The Well, one pair per trajectory, held in memory.

    Each item is a dictionary with
        ``input_fields``:     (1, *spatial, C) the initial condition u_0
        ``output_fields``:    (1, *spatial, C) the state u_horizon
        ``constant_scalars``: (num_parameters,) the physical parameters

    Args:
        path: Root of The Well datasets.
        dataset_name: e.g. ``"active_matter"``.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        horizon: Number of time steps between input and target.
    """

    def __init__(self, path: str, dataset_name: str, split: str, horizon: int):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        trajectories = WellDataset(
            well_base_path=path,
            well_dataset_name=dataset_name,
            well_split_name=split,
            n_steps_input=1,
            n_steps_output=horizon,
            full_trajectory_mode=True,
            max_rollout_steps=horizon,
            min_dt_stride=1,
            max_dt_stride=1,
            return_grid=False,
        )

        inputs, outputs, parameters = [], [], []
        for i in range(len(trajectories)):
            sample = trajectories[i]
            inputs.append(sample["input_fields"])                      # u_0
            outputs.append(sample["output_fields"][-1:].contiguous())  # u_1 .. u_horizon -> u_horizon
            parameters.append(sample["constant_scalars"])

        self.inputs = torch.stack(inputs)
        self.outputs = torch.stack(outputs)
        self.parameters = torch.stack(parameters)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_fields": self.inputs[idx],
            "output_fields": self.outputs[idx],
            "constant_scalars": self.parameters[idx],
        }
