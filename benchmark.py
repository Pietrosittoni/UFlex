"""Measure the parameter count, FLOPs, inference throughput, and training/inference cost of UFlex.

    python benchmark.py --config configs/active_matter/uflex.yaml --input-shape 1 256 256 11

``--input-shape`` is one sample without the batch axis: (T, *spatial, C).
"""

import argparse
import time
from typing import Dict

import torch
from torch.utils.flop_counter import FlopCounterMode

from uflex import UFlex
from uflex.config import Config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_flops(model: torch.nn.Module, x: torch.Tensor, parameters: torch.Tensor, with_backward: bool = True) -> int:
    """FLOPs of one forward (and backward) pass."""
    counter = FlopCounterMode(display=False)
    with counter:
        out = model(x, parameters=parameters)
        if with_backward:
            out.sum().backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


@torch.no_grad()
def inference_throughput(model, x, parameters, runs: int = 40, warmup: int = 10) -> float:
    """Samples per second in eval mode."""
    model.eval()
    for _ in range(warmup):
        model(x, parameters=parameters)
    synchronize(x.device)
    start = time.perf_counter()
    for _ in range(runs):
        model(x, parameters=parameters)
    synchronize(x.device)
    return runs * x.shape[0] / (time.perf_counter() - start)


def training_step_cost(model, x, parameters, mixed_precision: bool, accumulation_steps: int = 1,
                       steps: int = 10, warmup: int = 3) -> Dict[str, float]:
    """Average wall time and peak memory of one optimizer step (forward + backward + update)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    if x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(x.device)

    elapsed = 0.0
    for step in range(warmup + steps):
        synchronize(x.device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                loss = model(x, parameters=parameters).float().pow(2).mean() / accumulation_steps
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        synchronize(x.device)
        if step >= warmup:
            elapsed += time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated(x.device) / 1024**3 if x.device.type == "cuda" else float("nan")
    return {"train_step_time_s": elapsed / steps, "train_peak_memory_gb": peak}


@torch.no_grad()
def inference_cost(model, x, parameters, runs: int = 10) -> Dict[str, float]:
    """Average wall time and peak memory of one forward pass."""
    model.eval()
    if x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(x.device)
    synchronize(x.device)
    start = time.perf_counter()
    for _ in range(runs):
        model(x, parameters=parameters)
    synchronize(x.device)
    peak = torch.cuda.max_memory_allocated(x.device) / 1024**3 if x.device.type == "cuda" else float("nan")
    return {"inference_time_s": (time.perf_counter() - start) / runs, "inference_peak_memory_gb": peak}


def main(args) -> None:
    config = Config.from_yaml(args.config)
    device = torch.device(args.device or config.device)
    frames, *spatial, channels = args.input_shape
    num_parameters = config.model.get("num_parameters", 0)

    model = UFlex(in_channels=channels, **config.model).to(device)
    batch_size = args.batch_size or config.batch_size

    def inputs(batch: int):
        x = torch.randn(batch, frames, *spatial, channels, device=device)
        return x, torch.ones(batch, max(num_parameters, 1), device=device)

    results = {"parameters_M": sum(p.numel() for p in model.parameters()) / 1e6}
    if not args.skip_flops:
        results["gflops_fwd_bwd_per_sample"] = count_flops(model, *inputs(1)) / 1e9
    results["throughput_samples_per_s"] = inference_throughput(model, *inputs(args.throughput_batch_size))
    results.update(training_step_cost(model, *inputs(batch_size), mixed_precision=config.mixed_precision,
                                      accumulation_steps=config.accumulation_steps))
    results.update(inference_cost(model, *inputs(batch_size)))

    print(f"UFlex on input {tuple(args.input_shape)}, batch size {batch_size}:")
    for name, value in results.items():
        print(f"  {name:28s} {value:.4g}")

    if args.wandb:
        import wandb

        wandb.init(project=config.get("wandb_project", "UFlex") + "_benchmark", entity=config.get("wandb_entity"),
                   tags=config.get("tags"), config={**config.to_dict(), "input_shape": args.input_shape})
        wandb.log({f"benchmark/{k}": v for k, v in results.items()})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config (its `model` section is used).")
    parser.add_argument("--input-shape", type=int, nargs="+", required=True, metavar="N",
                        help="Shape of one sample: T, spatial sizes, C (e.g. 1 256 256 11).")
    parser.add_argument("--batch-size", type=int, help="Batch size for the training/inference cost (default: config).")
    parser.add_argument("--throughput-batch-size", type=int, default=32, help="Batch size for the throughput test.")
    parser.add_argument("--skip-flops", action="store_true", help="Skip FLOP counting (slow for large 3D inputs).")
    parser.add_argument("--wandb", action="store_true", help="Also log the results to Weights & Biases.")
    parser.add_argument("--device", help="Override the device, e.g. cuda:1.")
    main(parser.parse_args())
