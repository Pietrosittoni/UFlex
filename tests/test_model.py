"""Unit tests for the UFlex model: shapes, causal-in-time / global-in-space attention, and input checks."""

import pytest
import torch

from uflex import UFlex
from uflex.layers import block_causal_mask

SMALL = dict(depth=2, depth_res=1, depth_middle=1, heads=[4, 6], heads_middle=8, dim_head=16, downsampling_factor=4)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_block_causal_mask():
    mask = block_causal_mask(num_frames=3, tokens_per_frame=2, device="cpu")
    expected = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("shape, patch_size, out_channels", [
    ((2, 1, 64, 64, 3), (8, 8), None),          # 2D, single frame
    ((2, 3, 128, 64, 3), (8, 8), 5),            # 2D, several frames, other resolution and output channels
    ((1, 2, 32, 32, 32, 3), (8, 8, 8), None),   # 3D
    ((1, 1, 64, 16, 32, 3), (16, 8, 16), None), # 3D, anisotropic patches
    ((2, 2, 256, 3), (16,), None),              # 1D
])
def test_output_shape_follows_input(shape, patch_size, out_channels):
    model = UFlex(in_channels=3, out_channels=out_channels, patch_size=patch_size, **SMALL).eval()
    with torch.no_grad():
        y = model(torch.randn(*shape), torch.ones(shape[0], 1))
    assert y.shape == (shape[0], 1, *shape[2:-1], out_channels or 3)


def test_same_model_handles_any_resolution_and_number_of_frames():
    model = UFlex(in_channels=2, patch_size=(8, 8), **SMALL).eval()
    with torch.no_grad():
        for frames, height, width in [(1, 32, 128), (4, 64, 64), (2, 128, 256)]:
            y = model(torch.randn(1, frames, height, width, 2), torch.ones(1, 1))
            assert y.shape == (1, 1, height, width, 2)


@pytest.mark.parametrize("shape, patch_size, time_patch_size", [
    ((2, 4, 64, 64, 3), (8, 8), 1),
    ((2, 6, 64, 64, 3), (8, 8), 2),
    ((1, 3, 32, 32, 32, 3), (8, 8, 8), 1),
])
def test_attention_is_causal_in_time(shape, patch_size, time_patch_size):
    """Perturbing frame t changes the output at frames >= t and leaves earlier frames bit-identical."""
    model = UFlex(in_channels=3, patch_size=patch_size, time_patch_size=time_patch_size, num_parameters=2, **SMALL).eval()
    x, parameters = torch.randn(*shape), torch.rand(shape[0], 2)
    with torch.no_grad():
        reference = model._forward_all_frames(x, parameters)
        for t in range(0, shape[1], time_patch_size):
            perturbed = x.clone()
            perturbed[:, t:t + time_patch_size] += torch.randn_like(perturbed[:, t:t + time_patch_size])
            out = model._forward_all_frames(perturbed, parameters)
            assert torch.equal(out[:, :t], reference[:, :t]), f"frame {t} leaked into earlier frames"
            assert not torch.allclose(out[:, t:], reference[:, t:])


def test_attention_is_global_in_space():
    """A change in one corner of a frame reaches the opposite corner of the same frame."""
    model = UFlex(in_channels=3, patch_size=(8, 8), **SMALL).eval()
    x, parameters = torch.randn(1, 2, 64, 64, 3), torch.ones(1, 1)
    with torch.no_grad():
        reference = model._forward_all_frames(x, parameters)
        perturbed = x.clone()
        perturbed[:, :, -1, -1] += 1.0
        out = model._forward_all_frames(perturbed, parameters)
    assert not torch.allclose(out[:, :, 0, 0], reference[:, :, 0, 0])


def test_single_frame_attention_is_unmasked():
    """With one frame, causality is irrelevant: the result equals plain full attention."""
    import uflex.layers as layers

    model = UFlex(in_channels=3, patch_size=(8, 8), **SMALL).eval()
    x, parameters = torch.randn(2, 1, 64, 64, 3), torch.ones(2, 1)
    with torch.no_grad():
        reference = model(x, parameters)
        original, layers.block_causal_mask = layers.block_causal_mask, None  # would fail if it were called
        try:
            assert torch.equal(model(x, parameters), reference)
        finally:
            layers.block_causal_mask = original


def test_conditioning_on_physical_parameters():
    model = UFlex(in_channels=3, patch_size=(8, 8), num_parameters=2, **SMALL).eval()
    x = torch.randn(2, 1, 64, 64, 3)
    with torch.no_grad():
        a = model(x, torch.tensor([[0.1, 1.0], [0.1, 1.0]]))
        b = model(x, torch.tensor([[0.5, 1.0], [0.5, 1.0]]))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("shape, message", [
    ((2, 1, 60, 64, 3), "divisible by patch_size"),
    ((2, 1, 64, 64, 4), "input channels"),
    ((2, 1, 64, 64, 64, 3), "spatial axes"),
    ((2, 1, 32, 8, 3), "downsampling_factor"),
])
def test_invalid_inputs_raise_clear_errors(shape, message):
    model = UFlex(in_channels=3, patch_size=(8, 8), **SMALL)
    with pytest.raises(ValueError, match=message):
        model(torch.randn(*shape), torch.ones(shape[0], 1))


def test_frames_must_fill_time_patches():
    model = UFlex(in_channels=3, patch_size=(8, 8), time_patch_size=2, **SMALL)
    with pytest.raises(ValueError, match="time_patch_size"):
        model(torch.randn(1, 3, 64, 64, 3), torch.ones(1, 1))


def test_backward_under_bf16_autocast():
    model = UFlex(in_channels=3, patch_size=(8, 8), **SMALL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(torch.randn(2, 3, 64, 64, 3), torch.ones(2, 1)).float().pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_checkpoint_keys_are_stable():
    """Released checkpoints rely on these parameter names."""
    keys = set(UFlex(in_channels=3, patch_size=(8, 8), **SMALL).state_dict())
    for prefix in ["patch_norm.", "lift_conv.", "embed_norm.", "proj.", "timestep_embedder.0.mlp.",
                   "downsample_backbone.0.layers.0.attn.to_qkv.", "downsample_backbone.0.layers.0.mlp.fc3.",
                   "downsample_backbone.0.adaLN_modulation.1.", "down_project_condition.1.", "down_project_tokens.1.",
                   "middle_block.norm.", "upsample_backbone.1.layers.0.adaLN_modulation.1.",
                   "up_project_condition.0.", "up_project_tokens.1."]:
        assert any(k.startswith(prefix) for k in keys), prefix
