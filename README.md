# UFlex

Official PyTorch implementation of **UFlex** from the paper *&lt;Paper title&gt;*.

[**Paper**](#) | [**Citation**](#citation)

UFlex is a U-Net built entirely from transformer stages, designed as a neural surrogate for time-dependent PDEs. A single implementation handles 1D, 2D and 3D data, any resolution, any number of input frames, and conditioning on scalar physical parameters.

<p align="center">
<em>&lt;architecture figure&gt;</em>
</p>

## Highlights

- **Transformer U-Net.** The encoder merges groups of neighbouring tokens into fewer, wider tokens at each level. The decoder splits them back and adds skip connections. Every stage is a stack of adaLN-conditioned transformer blocks with rotary position embeddings and SwiGLU feed-forward layers.
- **Shape-agnostic.** Resolution, number of input frames and 2D vs. 3D come from the input tensor; nothing about the data shape is hard-coded in the model. Channel counts are read from the dataset when the model is built.
- **Causal in time, global in space.** With several input frames, each token attends to every token of its own frame and of all earlier frames, never to later ones. With a single frame this is ordinary full attention.
- **Physics conditioning.** Scalar parameters (Reynolds, Mach or Atwood numbers, ...) are embedded and modulate every block.

```
fields (B, T, *spatial, C)
  │ patchify + linear embedding                    tokens (B, T·P, d₀), ordered frame by frame
  ├─ level 0: transformer ─────────────────────────────── skip ──┐
  │  merge ×f                                                    │
  ├─ level 1: transformer ─────────────────────── skip ──┐       │
  │  merge ×f                                            │       │
  ├─ middle: transformer                                 │       │
  │  split ×f ── + ──────────────────────────────────────┘       │
  ├─ level 1: transformer                                        │
  │  split ×f ── + ──────────────────────────────────────────────┘
  ├─ level 0: transformer
  │ linear projection + unpatchify
next frame (B, 1, *spatial, C_out)
```

## Repository structure

```
uflex/
  model.py              UFlex model
  layers.py             attention with the block-causal mask, SwiGLU, adaLN transformer stages, parameter embedder
  engine.py             one-step training, validation and autoregressive rollout evaluation
  experiment.py         seeding, optimizer and schedule, epoch loop, checkpoints
  data/                 The Well, PDEBench compressible Navier–Stokes and steady-state datasets
configs/<benchmark>/    one YAML file per experiment
train.py                autoregressive training + rollout evaluation
finetune.py             fine-tuning from a pretrained backbone
train_steady_state.py   steady-state (input state -> output state) training
plot_rollout.py         plot a rollout against the reference
benchmark.py            parameters, FLOPs, throughput, memory
scripts/                dataset download helpers
tests/                  unit tests
```

## Installation

```bash
git clone https://github.com/Pietrosittoni/UFlex.git
cd UFlex
pip install -r requirements.txt
```

The code was tested with Python 3.11, PyTorch 2.7 and CUDA 12.6.

## Data

Every config reads its data from `data_path` (by default a folder under `data/`). Change it if your data lives elsewhere.

**The Well** (active_matter, planetswe, viscoelastic_instability, rayleigh_taylor_instability, convective_envelope_rsg):

```bash
the-well-download --base-path data/the_well --dataset active_matter
```

**PDEBench** compressible Navier–Stokes, 2D and 3D turbulent:

```bash
python scripts/download_pdebench.py --root_folder data/pdebench --pde_name 2d_cfd --pde_name 3d_cfd
```

**DiffusionPDE** steady-state problems (darcy, helmholtz, ...):

```bash
scripts/download_diffusionpde.sh data/diffusionpde darcy helmholtz
```

## Training

```bash
python train.py --config configs/active_matter/uflex.yaml
```

Training minimizes the one-step MSE on normalized fields. It uses AdamW, a linear warm-up followed by cosine decay, bfloat16 mixed precision and an EMA of the weights. At the end, `train.py` rolls the model out autoregressively on the test trajectories and reports the relative L2 error per step. It then saves `checkpoints/<run>/model.pt` together with the resolved `config.yaml`. Metrics are logged to Weights & Biases; set `WANDB_MODE=disabled` to turn logging off.

| Config | Data | Dim. | Resolution | Fields | Physical parameters |
|---|---|---|---|---|---|
| `configs/active_matter/uflex.yaml` | The Well · active_matter | 2D | 256 × 256 | 11 | 3 |
| `configs/planetswe/uflex.yaml` | The Well · planetswe | 2D | 256 × 512 | 3 | – |
| `configs/viscoelastic_instability/uflex.yaml` | The Well · viscoelastic_instability | 2D | 512 × 512 | 8 | 5 |
| `configs/cns_2d/uflex.yaml` | PDEBench · compressible Navier–Stokes | 2D | 512 × 512 | 4 | 1 |
| `configs/rayleigh_taylor_instability/uflex.yaml` | The Well · rayleigh_taylor_instability | 3D | 128³ | 4 | 1 |
| `configs/convective_envelope_rsg/uflex.yaml` | The Well · convective_envelope_rsg | 3D | 256 × 128 × 256 | 6 | – |
| `configs/cns_3d/uflex.yaml` | PDEBench · compressible Navier–Stokes | 3D | 64³ | 5 | – |

`uflex_small.yaml` variants are provided for active_matter and planetswe.

**Fine-tuning.** `finetune.py` initializes the backbone from `pretrained_checkpoint` and trains new input/output projections. The example config fine-tunes on 3D compressible Navier–Stokes with 4 input frames, so attention is causal in time:

```bash
python finetune.py --config configs/cns_3d_finetune/uflex.yaml
```

**Steady state.** Learn a direct map from one state to another, for example from the initial condition of active_matter to its state `horizon` steps later:

```bash
python train_steady_state.py --config configs/active_matter_steady/uflex.yaml
```

## Evaluation

Plot a rollout against the reference trajectory for one field:

```bash
python plot_rollout.py --config configs/planetswe/uflex.yaml --checkpoint checkpoints/<run>/model.pt --channel 0
```

Measure the model's cost on an input shape `(T, *spatial, C)`:

```bash
python benchmark.py --config configs/active_matter/uflex.yaml --input-shape 1 256 256 11
```

## Using UFlex in your own code

```python
import torch
from uflex import UFlex

model = UFlex(
    in_channels=4,
    patch_size=(8, 8),        # one entry per spatial axis: (8, 8) -> 2D, (8, 8, 8) -> 3D
    depth=2, heads=(8, 12), heads_middle=18, dim_head=64,
    num_parameters=1,         # scalar physical parameters used for conditioning
)

x = torch.randn(2, 4, 256, 256, 4)       # (batch, input frames, H, W, fields)
mach = torch.tensor([[0.1], [1.0]])      # (batch, num_parameters)
next_frame = model(x, mach)              # (2, 1, 256, 256, 4)
```

<details>
<summary><b>Model arguments</b></summary>

| Argument | Meaning |
|---|---|
| `in_channels`, `out_channels` | Number of input and predicted fields (`out_channels` defaults to `in_channels`). |
| `patch_size` | Patch size per spatial axis; its length sets the dimensionality. |
| `time_patch_size` | Consecutive frames grouped into one token. |
| `depth` | Number of encoder / decoder levels. |
| `depth_res`, `depth_middle` | Transformer blocks per level and in the middle stage. |
| `heads`, `heads_middle`, `dim_head` | Heads per level (finest to coarsest), heads in the middle stage, and head width. The width of level `i` is `heads[i] * dim_head`. |
| `downsampling_factor` | Tokens merged into one between two levels. |
| `num_parameters` | Number of conditioning scalars (0 for none). |
| `dropout`, `drop_path` | Dropout and maximum stochastic-depth rate. |
| `normalize_input` | Per-sample standardization of the input, undone on the output. |

Input constraints: every spatial size must be divisible by its patch size, the number of patches per frame must be divisible by `downsampling_factor ** depth`, and the number of frames must be a multiple of `time_patch_size`. The model raises a descriptive error otherwise.
</details>

## Tests

```bash
pytest tests
```

The tests cover output shapes in 1D, 2D and 3D and causality in time: perturbing a later frame leaves the outputs of earlier frames bit-identical. They also check global attention in space, the input checks, and that every released config builds and runs.

## Citation

If you use this code, please cite:

```bibtex
@article{uflex,
  title   = {<Paper title>},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {<Year>}
}
```

## Acknowledgements

The data pipelines build on [The Well](https://github.com/PolymathicAI/the_well), [PDEBench](https://github.com/pdebench/PDEBench) and the [DiffusionPDE](https://huggingface.co/datasets/jcy20/DiffusionPDE-normalized) normalized datasets. The conditioning follows the adaLN design of [DiT](https://github.com/facebookresearch/DiT), and rotary embeddings come from [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch).
