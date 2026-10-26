import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from solarwm.backends.wan22.ref2va import lora as lora_module
from solarwm.backends.wan22.ref2va.layout import build_ref2va_layout
from solarwm.backends.wan22.ref2va.model import Ref2VAWanModel
from solarwm.backends.wan22.runtime.modeling import model as official_module

TINY = dict(dim=48, ffn_dim=64, num_heads=4, num_layers=2)


def sdpa_attention(q, k, v, k_lens=None, window_size=(-1, -1), **_):
    del k_lens, window_size
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


@pytest.fixture
def official(monkeypatch):
    monkeypatch.setattr(official_module, "flash_attention", sdpa_attention)
    torch.manual_seed(0)
    model = official_module.WanModel(model_type="t2v", in_dim=16, out_dim=16, **TINY).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model


def ours_from(official, family="wan22_i2v_a14b", **extra):
    model = Ref2VAWanModel(family, in_dim=16, out_dim=16, **TINY, **extra).float().eval()
    missing, unexpected = model.load_state_dict(official.state_dict(), strict=False)
    assert not unexpected
    assert all(name.startswith("delta_embedding.") for name in missing)
    model.initialize_anyflow_delta_from_time()
    return model


def single_video_coords(frames, height, width):
    doc = build_ref2va_layout(
        picture_latent_sizes=((2, 2),),
        proxy_latent_size=(2, 2),
        proxy_frames=0,
        target_latent_size=(height, width),
        frame_range=(0, frames),
        teacher_forcing=False,
        chunk_latents=frames,
    )
    target = doc.segments[-1]
    coords = doc.coords[target.slice].copy()
    coords[:, 0] -= coords[0, 0]
    return torch.from_numpy(coords)


@pytest.mark.parametrize("per_frame_time", [False, True])
def test_matches_official_wan_forward(official, per_frame_time):
    ours = ours_from(official)
    frames, height, width = 3, 4, 6
    generator = torch.Generator().manual_seed(1)
    x = torch.randn((16, frames, height, width), generator=generator)
    context = torch.randn((7, 4096), generator=generator)
    tokens_per_frame = (height // 2) * (width // 2)
    seq_len = frames * tokens_per_frame
    if per_frame_time:
        per_token = torch.tensor([1.0, 400.0, 900.0]).repeat_interleave(tokens_per_frame)
        t_official = per_token[None]
    else:
        per_token = torch.full((seq_len,), 500.0)
        t_official = torch.tensor([500.0])
    with torch.no_grad():
        expected = official([x], t_official, [context], seq_len)[0]
        tokens = ours(
            [x[None]],
            coords=single_video_coords(frames, height, width),
            timesteps=per_token,
            context=context[None],
        )
        got = ours.unpatchify(tokens, frames, height // 2, width // 2)[0]
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_output_slice_and_segments_match_full(official):
    ours = ours_from(official)
    generator = torch.Generator().manual_seed(2)
    a = torch.randn((1, 16, 1, 4, 4), generator=generator)
    b = torch.randn((1, 16, 2, 4, 6), generator=generator)
    coords = torch.from_numpy(
        np.concatenate(
            [
                np.stack(np.meshgrid([0], range(2), range(2), indexing="ij"), -1).reshape(-1, 3),
                np.stack(np.meshgrid([1, 2], range(2), range(3), indexing="ij"), -1).reshape(-1, 3),
            ]
        )
    )
    times = torch.cat((torch.full((4,), 1.0), torch.full((12,), 700.0)))
    context = torch.randn((1, 5, 4096), generator=generator)
    with torch.no_grad():
        full = ours([a, b], coords=coords, timesteps=times, context=context)
        part = ours([a, b], coords=coords, timesteps=times, context=context, output=(4, 16))
        empty = ours([a, b], coords=coords, timesteps=times, context=context, output=(0, 0))
    torch.testing.assert_close(part, full[:, 4:16])
    assert empty.shape == (1, 0, full.shape[-1])


def test_anyflow_delta_starts_as_time_embedding(official):
    ours = ours_from(official, anyflow_gate=0.25)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn((1, 16, 2, 4, 4), generator=generator)
    coords = single_video_coords(2, 4, 4)
    t = torch.full((coords.shape[0],), 600.0)
    context = torch.randn((1, 3, 4096), generator=generator)
    with torch.no_grad():
        same = ours([x], coords=coords, timesteps=t, context=context, r_timesteps=t)
        diagonal = ours([x], coords=coords, timesteps=t, context=context)
        other = ours(
            [x], coords=coords, timesteps=t, context=context, r_timesteps=torch.full_like(t, 100.0)
        )
    torch.testing.assert_close(same, diagonal)
    assert not torch.allclose(same, other)


def test_lora_zero_init_merge_and_state(official):
    ours = ours_from(official)
    generator = torch.Generator().manual_seed(4)
    x = torch.randn((1, 16, 2, 4, 4), generator=generator)
    coords = single_video_coords(2, 4, 4)
    t = torch.full((coords.shape[0],), 300.0)
    context = torch.randn((1, 3, 4096), generator=generator)
    with torch.no_grad():
        base = ours([x], coords=coords, timesteps=t, context=context)
    names = lora_module.inject_lora(ours, target="main_attention_qkvo", rank=4, alpha=4)
    assert len(names) == 4 * TINY["num_layers"]
    for parameter in ours.parameters():
        parameter.requires_grad_(False)
    for module in ours.modules():
        if isinstance(module, lora_module.LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    with torch.no_grad():
        torch.testing.assert_close(ours([x], coords=coords, timesteps=t, context=context), base)
        for module in ours.modules():
            if isinstance(module, lora_module.LoRALinear):
                module.lora_B.normal_(0.0, 0.05, generator=generator)
        adapted = ours([x], coords=coords, timesteps=t, context=context)
    assert not torch.allclose(adapted, base)
    state = lora_module.adapter_state(ours)
    assert len(state) == 2 * len(names) and all(".lora_" in name for name in state)

    fresh = ours_from(official)
    lora_module.inject_lora(fresh, target="main_attention_qkvo", rank=4, alpha=4)
    for name, parameter in fresh.named_parameters():
        parameter.requires_grad_(".lora_" in name)
    assert lora_module.load_adapter_state(fresh, state) == len(state)
    assert lora_module.merge_lora(fresh) == len(names)
    assert not any(isinstance(m, lora_module.LoRALinear) for m in fresh.modules())
    with torch.no_grad():
        merged = fresh([x], coords=coords, timesteps=t, context=context)
    torch.testing.assert_close(merged, adapted, atol=1e-5, rtol=1e-5)
