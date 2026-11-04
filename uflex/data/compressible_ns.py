"""PDEBench compressible Navier–Stokes datasets (2D and 3D)."""

import glob
import os
import re
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = {
    2: ("Vx", "Vy", "density", "pressure"),
    3: ("Vx", "Vy", "Vz", "density", "pressure"),
}
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.2}  # the remaining 10% of trajectories is the test split


class CompressibleNSDataset(Dataset):
    """Compressible Navier–Stokes trajectories stored in PDEBench HDF5 files.

    Every file holds one array per field (see ``FIELDS``) of shape (samples, time, *spatial).
    Trajectories are split by sample index into train / val / test (70 / 20 / 10 %).

    Each item is a dictionary with
        ``input_fields``:     (n_steps_input, *spatial, C)
        ``output_fields``:    (n_steps_output, *spatial, C), or the full rollout target
                              when ``full_trajectory_mode`` is set
        ``constant_scalars``: (1,) the Mach number, parsed from the file name

    Args:
        base_path: Directory containing the ``.h5`` / ``.hdf5`` files.
        spatial_dims: 2 or 3.
        split: ``"train"``, ``"val"`` or ``"test"``.
        n_steps_input: Number of input frames.
        n_steps_output: Number of target frames (one-step training uses 1).
        stride: Temporal stride between consecutive frames.
        full_trajectory_mode: Return one item per trajectory: the first ``n_steps_input``
            frames as input and the next ``max_rollout_steps`` frames as target.
        max_rollout_steps: Length of the rollout target in ``full_trajectory_mode``.
        filename: If given, only use the file with this name.
    """

    def __init__(
        self,
        base_path: str,
        spatial_dims: int = 2,
        split: str = "train",
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        full_trajectory_mode: bool = False,
        max_rollout_steps: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        if spatial_dims not in FIELDS:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self.fields = FIELDS[spatial_dims]
        self.spatial_dims = spatial_dims
        self.split = split
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.stride = stride
        self.full_trajectory_mode = full_trajectory_mode
        self.max_rollout_steps = max_rollout_steps or n_steps_output

        files = sorted(glob.glob(os.path.join(base_path, "*.h5")) + glob.glob(os.path.join(base_path, "*.hdf5")))
        if filename is not None:
            files = [f for f in files if os.path.basename(f) == filename]
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {base_path}" + (f" named {filename}" if filename else ""))

        self.index = []  # (file, trajectory, time of the first target frame)
        for path in files:
            with h5py.File(path, "r") as f:
                num_samples, num_times = f[self.fields[0]].shape[:2]
            for sample in self._split_samples(num_samples):
                if full_trajectory_mode:
                    self.index.append((path, sample, 0))
                else:
                    first, last = n_steps_input * stride, num_times - n_steps_output * stride
                    self.index.extend((path, sample, t) for t in range(first, last))

    def _split_samples(self, num_samples: int) -> range:
        n_train = int(num_samples * SPLIT_FRACTIONS["train"])
        n_val = int(num_samples * SPLIT_FRACTIONS["val"])
        bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, num_samples)}
        return range(*bounds[self.split])

    @staticmethod
    def _mach_number(path: str) -> float:
        """Parse the Mach number from file names such as ``2D_CFD_Turb_M1.0_...`` (defaults to 1.0)."""
        match = re.search(r"M(\d+\.?\d*)", os.path.basename(path))
        return float(match.group(1)) if match else 1.0

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        path, sample, t = self.index[idx]
        with h5py.File(path, "r") as f:
            fields = np.stack([f[name][sample] for name in self.fields], axis=-1)  # (T, *spatial, C)

        if self.full_trajectory_mode:
            x = fields[: self.n_steps_input : self.stride]
            y = fields[self.n_steps_input : self.n_steps_input + self.max_rollout_steps : self.stride]
        else:
            x = fields[t - self.n_steps_input * self.stride : t : self.stride]
            y = fields[t : t + self.n_steps_output * self.stride : self.stride]

        # Fallbacks for degenerate windows, and make sure a time axis is always present.
        if x.shape[0] == 0:
            x = fields[None, t]
        if y.shape[0] == 0:
            y = fields[None, t + self.stride]
        if x.ndim == self.spatial_dims + 1:
            x = x[None]
        if y.ndim == self.spatial_dims + 1:
            y = y[None]

        return {
            "input_fields": torch.from_numpy(x).float(),
            "output_fields": torch.from_numpy(y).float(),
            "constant_scalars": torch.tensor([self._mach_number(path)], dtype=torch.float32),
        }
