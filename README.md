# Simplifying Transformer-Based U-Net Neural Physics Simulators

**Pietro Sittoni**<sup>1</sup>, **Francesco Tudisco**<sup>1,2,3</sup>

<sup>1</sup>Gran Sasso Science Institute, L'Aquila, Italy &nbsp; <sup>2</sup>University of Edinburgh, UK &nbsp; <sup>3</sup>Miniml.AI

*NeurIPS 2026* · [**Citation**](#citation)
<!-- TODO: add the arXiv / OpenReview link once available. -->

Official PyTorch implementation of **UFlex**, an attention-based U-Net for neural physics simulation. Its Transformer blocks stay close to the standard design: multi-head self-attention with RoPE, RMSNorm, a SwiGLU MLP and residual connections. They operate on a one-dimensional token sequence, so the same architecture applies to 1D, 2D, 3D and spatiotemporal regular-grid problems without design changes. UFlex reaches state-of-the-art accuracy on seven benchmarks from The Well and PDEBench, at resolutions up to 512 × 512 in 2D and 256 × 128 × 256 in 3D, while using less training memory and time than state-of-the-art Transformer baselines.

<p align="center">
  <img src="assets/uflex_architecture.png" width="100%" alt="UFlex architecture">
</p>
<p align="center"><em>Left: the UFlex U-Net (TF Blocks = Transformer blocks). Right: one Transformer block with adaRMSN-Zero conditioning.</em></p>

## Highlights

- **A U-Net on a 1D token sequence.** The state is cut into patches and flattened into one token sequence. Downsampling is a learnable linear map that merges `f = 4` consecutive tokens into one; upsampling is a learnable linear map that expands each token back into `f` tokens. Encoder and decoder features at the same scale are added through skip connections.
- **Standard Transformer blocks.** Multi-head self-attention with RoPE, a SwiGLU MLP and RMSNorm pre-normalization. There are no windowed, axial or Fourier attention layers.
- **Physics conditioning (adaRMSN-Zero).** Each scalar parameter (Mach number, Atwood number, viscosity, ...) is embedded by a small MLP. The embeddings are summed and modulate every block, as in adaLN-Zero but with RMSNorm.
- **One model for every dimensionality.** Resolution, number of input frames and 2D vs. 3D come from the input tensor. Temporal context is folded into the patches (`time_patch_size` frames per token), and a model trained in 2D can be fine-tuned directly in 3D.
- **Causal in time, global in space.** With several input frames, each token attends to every token of its own frame and of all earlier frames, never to later ones. With a single frame this is ordinary full attention.

## Results

Test error (nRMSE of the autoregressive rollout over the first 30% of the temporal horizon) of UFlex-B compared with the best baseline on each dataset. Lower is better; see the paper for all baselines, efficiency metrics and longer horizons.

| | Active Matter | Planetswe | Viscoelastic Inst. | Comp. NS 2D | Comp. NS 3D | Rayleigh–Taylor | Conv. Envelope RSG |
|---|---|---|---|---|---|---|---|
| Resolution | 256² | 256 × 512 | 512² | 512² | 64³ | 128³ | 256 × 128 × 256 |
| Best baseline | 0.459 (MPP) | 0.300 (Poseidon) | **0.091** (DPOT) | 0.091 (MPP) | 0.219 (DPOT) | 0.075 (U-Net) | 0.315 (U-Net) |
| **UFlex-B** | **0.343** | **0.243** | 0.100 | **0.087** | **0.197** | **0.040** | **0.256** |

UFlex-B also has the lowest training memory and the fastest training step of all compared models wherever these were measured (single A100).

## Repository structure

```
uflex/
  model.py              UFlex model
  layers.py             attention with the block-causal mask, SwiGLU MLP, adaRMSN-Zero Transformer stages
  engine.py             one-step training, validation and autoregressive rollout evaluation
  experiment.py         seeding, optimizer and schedule, epoch loop, checkpoints
  data/                 The Well, PDEBench compressible Navier–Stokes and static-prediction datasets
configs/<benchmark>/    uflex_s.yaml, uflex_b.yaml, uflex_l.yaml
train.py                autoregressive training + rollout evaluation
finetune.py             2D -> 3D transfer: fine-tuning from a pretrained backbone
train_steady_state.py   static prediction (input state -> target state)
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
python train.py --config configs/active_matter/uflex_b.yaml
```

Every benchmark folder holds one config per model size: `uflex_s.yaml`, `uflex_b.yaml` and `uflex_l.yaml`. UFlex-B is the model behind the main results, and the S and L configs inherit all data and optimization settings from it. Batch sizes, gradient accumulation and epochs follow Table 4 of the paper.

Training minimizes the one-step MSE on normalized fields. It uses AdamW (β₂ = 0.95, no weight decay) and a 1000-step linear warm-up to 5·10⁻⁴ followed by cosine decay to 10⁻⁶. It also uses bfloat16 mixed precision, gradient clipping at norm 1 and an EMA of the weights with decay 0.998.

At the end, `train.py` rolls the model out autoregressively on the test trajectories. It reports the test error over the first 30/45/60/75/90/100% of the horizon (TE in the paper) and saves `checkpoints/<run>/model.pt` with the resolved `config.yaml`. Metrics are logged to Weights & Biases; set `WANDB_MODE=disabled` to turn logging off.

| Benchmark (`configs/…`) | Data | Dim. | Resolution | Fields | Physical parameters | Input frames |
|---|---|---|---|---|---|---|
| `active_matter` | The Well · active_matter | 2D | 256 × 256 | 11 | 3 | 1 |
| `planetswe` | The Well · planetswe | 2D | 256 × 512 | 3 | – | 1 |
| `viscoelastic_instability` | The Well · viscoelastic_instability | 2D | 512 × 512 | 8 | 5 | 1 |
| `cns_2d` | PDEBench · compressible Navier–Stokes | 2D | 512 × 512 | 4 | 1 | 1 |
| `cns_3d` | PDEBench · compressible Navier–Stokes | 3D | 64³ | 5 | – | 4 |
| `rayleigh_taylor_instability` | The Well · rayleigh_taylor_instability | 3D | 128³ | 4 | 1 | 1 |
| `convective_envelope_rsg` | The Well · convective_envelope_rsg | 3D | 256 × 128 × 256 | 6 | – | 1 |

**Transfer from 2D to 3D.** `finetune.py` initializes the backbone from a 2D checkpoint (`pretrained_checkpoint`) and trains new input/output projections. Its example config fine-tunes UFlex-B on 3D compressible Navier–Stokes for 10 epochs (Table 2 of the paper):

```bash
python finetune.py --config configs/cns_3d_finetune/uflex_b.yaml
```

**Static prediction.** Learn a direct map from one state to another, for example from the initial condition of active_matter to its tenth time step (Section G.4):

```bash
python train_steady_state.py --config configs/active_matter_steady/uflex_b.yaml
```

## Evaluation

Plot a rollout against the reference trajectory for one field:

```bash
python plot_rollout.py --config configs/planetswe/uflex_b.yaml --checkpoint checkpoints/<run>/model.pt --channel 0
```

Measure the model's cost on an input shape `(T, *spatial, C)`:

```bash
python benchmark.py --config configs/active_matter/uflex_b.yaml --input-shape 1 256 256 11
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

**Model sizes from the paper.** All sizes use `dim_head = 64`, `downsampling_factor = 4` and patch size 8 (16 × 8 × 16 on Convective Envelope RSG). Parameter counts are for Planetswe; they vary slightly with the number of fields.

| Size | `heads` | `heads_middle` | `depth_res` | `depth_middle` | Parameters |
|---|---|---|---|---|---|
| UFlex-S | `[6, 8]` | 12 | 2 | 2 | 58M |
| UFlex-B | `[8, 12]` | 18 | 2 | 2 | 124M |
| UFlex-L | `[12, 16]` | 24 | 3 | 4 | 374M |

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
@inproceedings{sittoni2026uflex,
  title     = {Simplifying Transformer-Based U-Net Neural Physics Simulators},
  author    = {Sittoni, Pietro and Tudisco, Francesco},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

The benchmarks come from [The Well](https://github.com/PolymathicAI/the_well) and [PDEBench](https://github.com/pdebench/PDEBench); the steady-state loaders also support the [DiffusionPDE](https://huggingface.co/datasets/jcy20/DiffusionPDE-normalized) normalized datasets. The conditioning adapts adaLN-Zero from [DiT](https://github.com/facebookresearch/DiT), and rotary embeddings come from [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch).
