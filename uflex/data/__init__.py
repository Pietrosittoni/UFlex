"""Data loaders for every benchmark used in the paper.

All datasets yield dictionaries with ``input_fields`` (T_in, *spatial, C),
``output_fields`` (T_out, *spatial, C) and ``constant_scalars`` (num_parameters,).
"""

from typing import Tuple

from torch.utils.data import DataLoader

from uflex.data.compressible_ns import CompressibleNSDataset

# PDEBench compressible Navier–Stokes: dataset name -> (spatial dims, file to use).
COMPRESSIBLE_NS = {
    "CNS2d": (2, None),
    "CNS3d": (3, "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"),
}


def _loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders for autoregressive training.

    Returns:
        train: one-step pairs (``input_steps`` frames -> next frame), shuffled.
        val: one-step pairs from the test split.
        rollout: full test trajectories (``input_steps`` frames -> next ``rollout_steps`` frames).
    """
    if config.dataset_name in COMPRESSIBLE_NS:
        spatial_dims, filename = COMPRESSIBLE_NS[config.dataset_name]
        common = dict(
            base_path=config.data_path,
            spatial_dims=spatial_dims,
            n_steps_input=config.input_steps,
            stride=config.stride,
            filename=filename,
        )
        train = CompressibleNSDataset(split="train", **common)
        val = CompressibleNSDataset(split="test", **common)
        rollout = CompressibleNSDataset(
            split="test", n_steps_output=config.rollout_steps, full_trajectory_mode=True,
            max_rollout_steps=config.rollout_steps, **common,
        )
        if spatial_dims == 3:  # 3D trajectories are large: smaller evaluation batches
            val_batch_size, rollout_batch_size = config.batch_size // 2, 1
        else:
            val_batch_size = rollout_batch_size = config.batch_size
    else:
        from the_well.data import WellDataset

        common = dict(
            well_base_path=config.data_path,
            well_dataset_name=config.dataset_name,
            n_steps_input=config.input_steps,
            min_dt_stride=config.stride,
        )
        train = WellDataset(well_split_name="train", n_steps_output=1, return_grid=False, **common)
        val = WellDataset(well_split_name="test", n_steps_output=1, return_grid=False, **common)
        rollout = WellDataset(
            well_split_name="test", n_steps_output=config.rollout_steps, full_trajectory_mode=True,
            max_rollout_steps=config.rollout_steps, **common,
        )
        val_batch_size = rollout_batch_size = config.batch_size // 6

    return (
        _loader(train, config.batch_size, shuffle=True),
        _loader(val, val_batch_size, shuffle=False),
        _loader(rollout, rollout_batch_size, shuffle=False),
    )


def build_static_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Loaders for the static prediction task: initial condition -> state ``config.horizon`` steps later."""
    from uflex.data.static import StaticPredictionDataset

    common = dict(path=config.data_path, dataset_name=config.dataset_name, horizon=config.horizon)
    train = StaticPredictionDataset(split="train", **common)
    val = StaticPredictionDataset(split="test", **common)
    return _loader(train, config.batch_size, shuffle=True), _loader(val, config.batch_size // 2, shuffle=False)
