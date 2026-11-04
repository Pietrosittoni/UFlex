"""Building blocks of UFlex: normalization, attention, feed-forward and conditioned transformer stages.

Attribute names (``to_qkv``, ``adaLN_modulation``, ``mlp``, ...) are kept stable so that
checkpoints trained with earlier versions of the code load without any key remapping.
"""

import math
from typing import Optional

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.layers import DropPath
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Adaptive LayerNorm modulation: per-sample shift/scale broadcast over the token axis."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def block_causal_mask(num_frames: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    """Attention mask that is causal in time and dense in space.

    Tokens are ordered frame by frame. A token may attend to every token of its own frame
    (full spatial attention) and to every token of earlier frames, never to later frames.

    Returns:
        Boolean tensor of shape (N, N) with N = num_frames * tokens_per_frame,
        ``True`` where attention is allowed (the convention of ``scaled_dot_product_attention``).
    """
    frame_of_token = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    return frame_of_token[:, None] >= frame_of_token[None, :]


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension, with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * self.scale / rms


class Attention(nn.Module):
    """Multi-head self-attention with rotary position embeddings over the token sequence."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: bool = False):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(0.1 if dropout else 0.0)
        self.rotary_emb = RotaryEmbedding(dim=dim_head // 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.dropout(self.to_out(out))


class FeedForward(nn.Module):
    """SwiGLU feed-forward layer (Shazeer, 2020: https://arxiv.org/abs/2002.05202)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: bool = False):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)  # keep the parameter count of a plain MLP despite the gate
        self.fc1 = nn.Linear(dim, hidden_dim)  # activated branch
        self.fc2 = nn.Linear(hidden_dim, dim)  # output
        self.fc3 = nn.Linear(dim, hidden_dim)  # linear (gating) branch
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.1 if dropout else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x)) * self.fc3(x)))


class ConditionedTransformerBlock(nn.Module):
    """Pre-norm transformer block whose norms are modulated by a conditioning vector (adaLN-Zero style)."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: bool = False, drop_path: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, hidden_dim=4 * dim, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        h = self.attn(modulate(self.norm(x), shift_attn, scale_attn), attn_mask=attn_mask)
        x = x + self.drop_path(gate_attn.unsqueeze(1) * h)

        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * h)
        return x


class ConditionedTransformer(nn.Module):
    """A stack of conditioned transformer blocks followed by a modulated output norm.

    When the sequence holds more than one frame, attention is block-causal in time
    (see :func:`block_causal_mask`); with a single frame it is plain full attention.
    """

    def __init__(self, depth: int, heads: int, dim_head: int, dropout: bool = False, drop_path: float = 0.0):
        super().__init__()
        dim = heads * dim_head
        drop_path_rates = torch.linspace(0, drop_path, depth).tolist()  # stochastic depth grows with depth
        self.layers = nn.ModuleList(
            ConditionedTransformerBlock(dim, heads, dim_head, dropout=dropout, drop_path=rate)
            for rate in drop_path_rates
        )
        self.norm = RMSNorm(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        """
        Args:
            x: Tokens of shape (B, N, dim), ordered frame by frame (N = num_frames * tokens_per_frame).
            c: Conditioning vector of shape (B, dim).
            num_frames: Number of frames (time tokens) contained in the sequence.
        """
        attn_mask = None
        if num_frames > 1:
            attn_mask = block_causal_mask(num_frames, x.shape[1] // num_frames, x.device)

        for layer in self.layers:
            x = layer(x, c, attn_mask=attn_mask)

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return modulate(self.norm(x), shift, scale)


class ScalarEmbedder(nn.Module):
    """Embeds a scalar (e.g. a physical parameter) with sinusoidal features followed by an MLP.

    Same construction as the DiT timestep embedder.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
        """Map a (B,) tensor of scalars to (B, dim) cos/sin features at geometrically spaced frequencies."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(t, self.frequency_embedding_size))
