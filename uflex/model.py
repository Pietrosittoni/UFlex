"""UFlex: an attention-based U-Net neural physics simulator operating on a 1D token sequence.

Reference: P. Sittoni and F. Tudisco, "Simplifying Transformer-Based U-Net Neural Physics
Simulators", NeurIPS 2026.

Overview of one forward pass::

    fields (B, T, *spatial, C)
      -> patchify          tokens (B, N, patch_dim)        N = frames * patches_per_frame
      -> embed             tokens (B, N, dim_0)
      -> encoder           per level: transformer stage, keep skip, merge `downsampling_factor`
                           neighbouring tokens into one (fewer, wider tokens)
      -> middle stage      transformer stage at the coarsest level
      -> decoder           per level: split every token back into `downsampling_factor` tokens,
                           add the skip connection, transformer stage
      -> project + unpatchify  fields (B, T, *spatial, C_out)
      -> keep the last frame   next-step prediction (B, 1, *spatial, C_out)

Every stage is conditioned (adaLN) on an embedding of the scalar physical parameters.

Input shape is detected at run time: the same model accepts any number of input frames
and any spatial resolution compatible with the patch size. Whether the model is 1D, 2D or
3D is fixed by the number of entries in ``patch_size``.

Attention is full over space and causal over time: a token sees every token of its own
frame and of earlier frames, never of later frames. With a single input frame this is
ordinary full attention.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange
from torch import nn

from uflex.layers import ConditionedTransformer, RMSNorm, ScalarEmbedder

SPATIAL_AXES = ("h", "w", "d")


def merge_tokens(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Concatenate each group of `factor` consecutive tokens: (B, N, D) -> (B, N / factor, factor * D)."""
    return rearrange(x, "b (n f) d -> b n (f d)", f=factor)


def split_tokens(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Inverse of :func:`merge_tokens`: (B, N, D) -> (B, N * factor, D / factor)."""
    return rearrange(x, "b n (f d) -> b (n f) d", f=factor)


class UFlex(nn.Module):
    """UFlex model.

    The defaults correspond to UFlex-B from the paper (123M parameters); UFlex-S uses
    ``heads=(6, 8), heads_middle=12``.

    Args:
        in_channels: Number of physical fields in the input.
        out_channels: Number of physical fields to predict. Defaults to ``in_channels``.
        patch_size: Patch size along each spatial axis, e.g. ``(8, 8)`` for 2D data or
            ``(16, 8, 16)`` for 3D data. Its length sets the spatial dimensionality.
        time_patch_size: Number of consecutive frames grouped into one token.
        depth: Number of encoder (and decoder) levels.
        depth_res: Transformer blocks per encoder/decoder level.
        depth_middle: Transformer blocks in the middle stage.
        heads: Attention heads at each level, from finest to coarsest (length ``depth``).
            The token width of a level is ``heads[i] * dim_head``.
        heads_middle: Attention heads in the middle stage.
        dim_head: Width of one attention head.
        downsampling_factor: Number of tokens merged into one between two levels.
        num_parameters: Number of scalar physical parameters used for conditioning.
            With 0, the model expects a dummy tensor of ones of shape (B, 1).
        dropout: Enable dropout (p = 0.1) in attention and feed-forward layers.
        drop_path: Maximum stochastic-depth rate inside each transformer stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        patch_size: Sequence[int] = (8, 8),
        time_patch_size: int = 1,
        depth: int = 2,
        depth_res: int = 2,
        depth_middle: int = 2,
        heads: Sequence[int] = (8, 12),
        heads_middle: int = 18,
        dim_head: int = 64,
        downsampling_factor: int = 4,
        num_parameters: int = 0,
        dropout: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if len(heads) != depth:
            raise ValueError(f"`heads` needs one entry per level ({depth}), got {list(heads)}")
        if not 1 <= len(patch_size) <= len(SPATIAL_AXES):
            raise ValueError(f"`patch_size` needs one entry per spatial axis (1 to 3), got {list(patch_size)}")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.patch_size = tuple(patch_size)
        self.time_patch_size = time_patch_size
        self.depth = depth
        self.downsampling_factor = downsampling_factor

        # Token width at every level, finest to coarsest, plus the middle stage.
        self.level_dims = [h * dim_head for h in heads]
        self.middle_dim = heads_middle * dim_head
        for dim in self.level_dims[1:] + [self.middle_dim]:
            if dim % downsampling_factor:
                raise ValueError(f"Token width {dim} must be divisible by downsampling_factor={downsampling_factor}")

        voxels_per_patch = time_patch_size * math.prod(self.patch_size)
        patch_dim_in = voxels_per_patch * in_channels
        patch_dim_out = voxels_per_patch * self.out_channels
        stage = dict(dim_head=dim_head, dropout=dropout, drop_path=drop_path)
        coarser_dims = self.level_dims[1:] + [self.middle_dim]  # width of the level below each level

        # Patch embedding (fields -> tokens) and output projection (tokens -> fields).
        # Module names are kept from the original code base so released checkpoints load as-is.
        self.patch_norm = RMSNorm(patch_dim_in)
        self.lift_conv = nn.Linear(patch_dim_in, self.level_dims[0])  # linear patch embedding
        self.embed_norm = RMSNorm(self.level_dims[0])
        self.proj = nn.Linear(self.level_dims[0], patch_dim_out)

        # One embedder per physical parameter; their embeddings are summed.
        self.timestep_embedder = nn.ModuleList(
            ScalarEmbedder(self.level_dims[0]) for _ in range(max(num_parameters, 1))
        )

        # Encoder: a transformer stage per level, then merge tokens and widen to the next level.
        self.downsample_backbone = nn.ModuleList(
            ConditionedTransformer(depth_res, heads[i], **stage) for i in range(depth)
        )
        self.down_project_condition = nn.ModuleList(
            nn.Linear(dim, next_dim) for dim, next_dim in zip(self.level_dims, coarser_dims)
        )
        self.down_project_tokens = nn.ModuleList(
            nn.Linear(downsampling_factor * dim, next_dim) for dim, next_dim in zip(self.level_dims, coarser_dims)
        )

        self.middle_block = ConditionedTransformer(depth_middle, heads_middle, **stage)

        # Decoder (coarsest to finest): split tokens, narrow to the level width, add the skip, transformer stage.
        self.upsample_backbone = nn.ModuleList(
            ConditionedTransformer(depth_res, heads[i], **stage) for i in reversed(range(depth))
        )
        self.up_project_condition = nn.ModuleList(
            nn.Linear(coarser_dims[i], self.level_dims[i]) for i in reversed(range(depth))
        )
        self.up_project_tokens = nn.ModuleList(
            nn.Linear(coarser_dims[i] // downsampling_factor, self.level_dims[i]) for i in reversed(range(depth))
        )

    # ------------------------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Predict the frame that follows the input sequence.

        Args:
            x: Input fields of shape (B, T, *spatial, in_channels), e.g. (B, T, H, W, C) in 2D
               or (B, T, H, W, D, C) in 3D. T is the number of input frames (any multiple of
               ``time_patch_size``).
            parameters: Physical parameters of shape (B, num_parameters).

        Returns:
            Prediction of shape (B, 1, *spatial, out_channels).
        """
        return self._forward_all_frames(x, parameters)[:, -1:]

    def _forward_all_frames(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Run the network and return the output at every input frame, (B, T, *spatial, out_channels).

        Thanks to causal attention, the output at frame t depends only on frames <= t.
        """
        self._check_input(x)

        tokens, grid = self._patchify(x)
        num_frames = grid[0]
        tokens = self.embed_norm(self.lift_conv(self.patch_norm(tokens)))
        condition = self._embed_parameters(parameters)

        tokens, skips, condition = self._encode(tokens, condition, num_frames)
        tokens = self.middle_block(tokens, condition, num_frames)
        tokens = self._decode(tokens, condition, skips, num_frames)

        return self._unpatchify(self.proj(tokens), grid)

    def _embed_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Sum the embeddings of every physical parameter: (B, num_parameters) -> (B, dim_0)."""
        return sum(embed(parameters[:, i]) for i, embed in enumerate(self.timestep_embedder))

    def _encode(
        self, x: torch.Tensor, condition: torch.Tensor, num_frames: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        skips = []
        for stage, project_tokens, project_condition in zip(
            self.downsample_backbone, self.down_project_tokens, self.down_project_condition
        ):
            x = stage(x, condition, num_frames)
            skips.append(x)
            x = project_tokens(merge_tokens(x, self.downsampling_factor))
            condition = project_condition(condition)
        return x, skips, condition

    def _decode(
        self, x: torch.Tensor, condition: torch.Tensor, skips: List[torch.Tensor], num_frames: int
    ) -> torch.Tensor:
        for stage, project_tokens, project_condition in zip(
            self.upsample_backbone, self.up_project_tokens, self.up_project_condition
        ):
            x = project_tokens(split_tokens(x, self.downsampling_factor)) + skips.pop()
            condition = project_condition(condition)
            x = stage(x, condition, num_frames)
        return x

    # ------------------------------------------------------------------------------------
    # Shapes: patchify / unpatchify and input checks
    # ------------------------------------------------------------------------------------

    def _patch_patterns(self) -> Tuple[str, str]:
        """einops patterns for fields and tokens, e.g. in 2D:

        fields: "b (t pt) (h ph) (w pw) c"
        tokens: "b (t h w) (pt ph pw c)"

        Tokens are ordered frame by frame (time is the slowest axis), which is what
        lets token merging and the causal mask operate within frames.
        """
        axes = SPATIAL_AXES[: len(self.patch_size)]
        fields = "b (t pt) " + " ".join(f"({a} p{a})" for a in axes) + " c"
        tokens = "b (t " + " ".join(axes) + ") (pt " + " ".join(f"p{a}" for a in axes) + " c)"
        return fields, tokens

    def _patch_sizes(self) -> dict:
        axes = SPATIAL_AXES[: len(self.patch_size)]
        return {"pt": self.time_patch_size, **{f"p{a}": p for a, p in zip(axes, self.patch_size)}}

    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """(B, T, *spatial, C) -> tokens (B, N, patch_dim), plus the token grid (frames, *patches_per_axis)."""
        frames = x.shape[1] // self.time_patch_size
        patches = [size // p for size, p in zip(x.shape[2:-1], self.patch_size)]
        fields, tokens = self._patch_patterns()
        return rearrange(x, f"{fields} -> {tokens}", **self._patch_sizes()), (frames, *patches)

    def _unpatchify(self, x: torch.Tensor, grid: Tuple[int, ...]) -> torch.Tensor:
        """tokens (B, N, patch_dim) -> (B, T, *spatial, C), using the grid returned by :meth:`_patchify`."""
        fields, tokens = self._patch_patterns()
        axes = SPATIAL_AXES[: len(self.patch_size)]
        grid_sizes = {"t": grid[0], **dict(zip(axes, grid[1:]))}
        return rearrange(x, f"{tokens} -> {fields}", **grid_sizes, **self._patch_sizes())

    def _check_input(self, x: torch.Tensor) -> None:
        spatial_dims = len(self.patch_size)
        if x.ndim != spatial_dims + 3:
            raise ValueError(
                f"Expected input of shape (batch, time, {spatial_dims} spatial axes, channels) "
                f"for patch_size={self.patch_size}, got {tuple(x.shape)}"
            )
        frames, *spatial, channels = x.shape[1:]
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        if frames % self.time_patch_size:
            raise ValueError(f"Number of input frames ({frames}) must be a multiple of time_patch_size={self.time_patch_size}")
        if any(size % p for size, p in zip(spatial, self.patch_size)):
            raise ValueError(f"Spatial shape {tuple(spatial)} must be divisible by patch_size={self.patch_size}")

        # Tokens are merged within each frame only, so every level must hold a whole number of tokens per frame.
        patches_per_frame = math.prod(size // p for size, p in zip(spatial, self.patch_size))
        if patches_per_frame % self.downsampling_factor**self.depth:
            raise ValueError(
                f"Each frame has {patches_per_frame} patches, which must be divisible by "
                f"downsampling_factor**depth = {self.downsampling_factor**self.depth}"
            )
