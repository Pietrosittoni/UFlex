# Simplifying Transformer-Based U-Net Neural Physics Simulators

**Pietro Sittoni**<sup>1</sup>, **Francesco Tudisco**<sup>1,2,3</sup><br>
<sup>1</sup>Gran Sasso Science Institute &nbsp; <sup>2</sup>University of Edinburgh &nbsp; <sup>3</sup>Miniml.AI

*NeurIPS 2026* · [Citation](#citation)
<!-- TODO: add the arXiv / OpenReview link once available. -->

Official PyTorch implementation of **UFlex**, an attention-based U-Net that operates on a 1D token sequence. The same model handles 2D, 3D and spatiotemporal regular-grid problems.

<p align="center">
  <img src="assets/uflex_architecture.png" width="100%" alt="UFlex architecture">
</p>

- **U-Net on tokens:** learnable down/upsampling merges or splits `f = 4` consecutive tokens; additive skip connections.
- **Standard Transformer blocks:** self-attention with RoPE, SwiGLU MLP, RMSNorm.
- **Physics conditioning:** scalar parameters are embedded and injected with adaRMSN-Zero.
- **Shape-agnostic:** resolution, number of input frames and 2D vs. 3D are read from the input.
- **Causal in time, global in space:** with several input frames, tokens attend only to their own frame and earlier frames.

## Installation

```bash
git clone https://github.com/Pietrosittoni/UFlex.git
cd UFlex
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.7 and CUDA 12.6.

## Data

| Config | Dataset | Resolution | Source |
|---|---|---|---|
| `active_matter` | Active Matter (AM) | 256 × 256 | The Well |
| `planetswe` | Planetswe (PSWE) | 256 × 512 | The Well |
| `viscoelastic_instability` | Viscoelastic Instability (VI) | 512 × 512 | The Well |
| `cns_2d` | Compressible Navier–Stokes (CNS) | 512 × 512 | PDEBench |
| `cns_3d` | Compressible Navier–Stokes (CNS) | 64³ | PDEBench |
| `rayleigh_taylor_instability` | Rayleigh–Taylor Instability (RTI) | 128³ | The Well |
| `convective_envelope_rsg` | Convective Envelope RSG (CE-RGS) | 256 × 128 × 256 | The Well |

Configs read the data from `data_path`, which points under `data/` by default:

```bash
# The Well (one dataset at a time; the 3D ones are several hundred GB)
the-well-download --base-path data/the_well --dataset active_matter

# PDEBench
DARUS=https://darus.uni-stuttgart.de/api/access/datafile
CNS2D=data/pdebench/2D/CFD/2D_Train_Turb
curl -L --create-dirs -o $CNS2D/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 $DARUS/164685
curl -L --create-dirs -o $CNS2D/2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 $DARUS/164686
curl -L --create-dirs -o data/pdebench/3D/Train/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5 $DARUS/164694
```

## Experiments

| Experiment | Command |
|---|---|
| Main results and model scaling | `python train.py --config configs/<dataset>/uflex_{s,b,l}.yaml` |
| Transfer from 2D to 3D | `python finetune.py --config configs/cns_3d_finetune/uflex_b.yaml` |
| Static prediction | `python train_static.py --config configs/active_matter_static/uflex_b.yaml` |

- `train.py` trains with one-step teacher forcing, then evaluates autoregressive rollouts on the test set and reports TE at 30, 45, 60, 75, 90 and 100% of the horizon.
- `finetune.py` initializes the backbone from a 2D checkpoint; set `pretrained_checkpoint` in the config.
- `train_static.py` predicts the 10th time step of Active Matter directly from the initial condition.

Checkpoints are saved to `checkpoints/<run>/` together with the config. Use `--device cuda:1` to pick a GPU. Logging uses Weights & Biases; set `WANDB_MODE=disabled` to turn it off.

## Usage

```python
import torch
from uflex import UFlex

model = UFlex(in_channels=4, patch_size=(8, 8), num_parameters=1)  # (8, 8, 8) for 3D

x = torch.randn(2, 4, 256, 256, 4)       # (batch, frames, H, W, fields)
params = torch.tensor([[0.1], [1.0]])    # (batch, num_parameters)
next_frame = model(x, params)            # (2, 1, 256, 256, 4)
```

| Size | `heads` | `heads_middle` | `depth_res` | `depth_middle` | Parameters |
|---|---|---|---|---|---|
| UFlex-S | `[6, 8]` | 12 | 2 | 2 | 57M |
| UFlex-B (default) | `[8, 12]` | 18 | 2 | 2 | 123M |
| UFlex-L | `[12, 16]` | 24 | 3 | 4 | 374M |

Each spatial size must be divisible by the patch size, and the patches per frame must be divisible by `downsampling_factor ** depth`. All arguments are documented in [`uflex/model.py`](uflex/model.py).

## Citation

```bibtex
@inproceedings{sittoni2026uflex,
  title     = {Simplifying Transformer-Based U-Net Neural Physics Simulators},
  author    = {Sittoni, Pietro and Tudisco, Francesco},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

Data from [The Well](https://github.com/PolymathicAI/the_well) and [PDEBench](https://github.com/pdebench/PDEBench). Conditioning adapted from [DiT](https://github.com/facebookresearch/DiT); rotary embeddings from [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch).
