import numpy as np
import pytest
import torch

from solarwm.backends.wan22.ref2va.attention import TILE, build_window_mask, proxy_chunk_spans
from solarwm.backends.wan22.ref2va.layout import (
    CLEAN_ROLE,
    NOISY_ROLE,
    PAD_ROLE,
    PICTURE_ROLE,
    PROXY_ROLE,
    build_ref2va_layout,
    dense_window_mask,
)


def layout(**overrides):
    kwargs = dict(
        picture_latent_sizes=((4, 6), (2, 4)),
        proxy_latent_size=(2, 4),
        proxy_frames=5,
        target_latent_size=(4, 4),
        frame_range=(0, 6),
        teacher_forcing=True,
        chunk_latents=2,
    )
    kwargs.update(overrides)
    return build_ref2va_layout(**kwargs)


def test_bidirectional_layout_runs_one_clock():
    doc = layout(teacher_forcing=False, frame_range=(0, 5))
    roles = [segment.role for segment in doc.segments]
    assert roles == [PICTURE_ROLE, PICTURE_ROLE, PROXY_ROLE, NOISY_ROLE]
    picture, oc, proxy, target = doc.segments
    assert doc.coords[picture.slice, 0].tolist() == [0] * picture.length
    assert set(doc.coords[oc.slice, 0].tolist()) == {1}
    assert doc.coords[proxy.start, 0] == 2 and doc.coords[proxy.stop - 1, 0] == 2 + 4
    assert doc.coords[target.start, 0] == 2 + 5 and doc.coords[target.stop - 1, 0] == 2 + 5 + 4
    # Every segment's spatial grid starts at zero.
    for segment in doc.segments:
        assert doc.coords[segment.start, 1:].tolist() == [0, 0]
        assert doc.coords[segment.stop - 1, 1:].tolist() == [segment.height - 1, segment.width - 1]
    assert doc.chunks[target.slice].reshape(5, -1)[:, 0].tolist() == [0, 0, 1, 1, 2]


def test_teacher_forcing_copies_share_coordinates():
    doc = layout()
    clean, noisy = doc.only(CLEAN_ROLE), doc.only(NOISY_ROLE)
    assert np.array_equal(doc.coords[clean.slice], doc.coords[noisy.slice])
    # The proxy stops at its cached length; the target runs on to the padded rollout length.
    assert doc.only(PROXY_ROLE).frames == 5 and noisy.frames == 6


def test_noisy_first_window_document():
    doc = layout(frame_range=(2, 6), noisy_first=4)
    clean, noisy = doc.only(CLEAN_ROLE), doc.only(NOISY_ROLE)
    assert (clean.first_frame, clean.frames) == (2, 2)
    assert (noisy.first_frame, noisy.frames) == (4, 2)
    full = layout()
    assert (
        doc.coords[noisy.start, 0]
        == full.coords[full.only(NOISY_ROLE).start + 4 * noisy.tokens_per_frame, 0]
    )


def test_window_visibility_rules():
    doc = layout()
    dense = dense_window_mask(doc, window_chunks=2, windowed_clean=True)
    clean, noisy, proxy = doc.only(CLEAN_ROLE), doc.only(NOISY_ROLE), doc.only(PROXY_ROLE)
    picture = doc.segments[0]

    def rows(segment, frame):
        start = segment.start + (frame - segment.first_frame) * segment.tokens_per_frame
        return slice(start, start + segment.tokens_per_frame)

    assert (
        dense[picture.slice][:, picture.slice].all()
        and not dense[picture.slice][:, proxy.slice].any()
    )
    # Noisy chunk 2 (latents 4, 5) reads clean chunk 1, not clean chunk 0 nor itself clean.
    assert dense[rows(noisy, 4)][:, rows(clean, 2)].all()
    assert not dense[rows(noisy, 4)][:, rows(clean, 0)].any()
    assert not dense[rows(noisy, 4)][:, rows(clean, 4)].any()
    assert dense[rows(noisy, 4)][:, rows(noisy, 5)].all()
    assert not dense[rows(noisy, 4)][:, rows(noisy, 2)].any()
    # Proxy window: chunks 1 and 2 only.
    assert (
        dense[rows(noisy, 4)][:, rows(proxy, 2)].all()
        and not dense[rows(noisy, 4)][:, rows(proxy, 0)].any()
    )
    # Windowed clean history drops chunk 0 from clean chunk 2; unwindowed keeps it.
    assert not dense[rows(clean, 4)][:, rows(clean, 0)].any()
    full_history = dense_window_mask(doc, window_chunks=2, windowed_clean=False)
    assert full_history[rows(clean, 4)][:, rows(clean, 0)].all()


def test_proxy_chunk_spans_cover_proxy():
    doc = layout()
    spans = proxy_chunk_spans(doc)
    proxy = doc.only(PROXY_ROLE)
    assert spans[0][0] == proxy.start and spans[-1][1] == proxy.stop
    assert [b - a for a, b in spans] == [
        2 * proxy.tokens_per_frame,
        2 * proxy.tokens_per_frame,
        proxy.tokens_per_frame,
    ]


@pytest.mark.parametrize("windowed_clean", [False, True])
def test_block_mask_matches_dense(windowed_clean):
    pytest.importorskip("torch.nn.attention.flex_attention")
    doc = layout(target_latent_size=(8, 16), proxy_latent_size=(4, 8), frame_range=(0, 6))
    dense = dense_window_mask(doc, window_chunks=2, windowed_clean=windowed_clean)
    mask = build_window_mask(doc, window_chunks=2, windowed_clean=windowed_clean, device="cpu")
    padded = mask.padded_length
    expected = np.zeros((padded, padded), dtype=bool)
    expected[: doc.length, : doc.length] = dense
    pad = np.arange(padded) >= doc.length
    expected |= pad[:, None] & pad[None, :]

    q = torch.arange(padded)[:, None]
    k = torch.arange(padded)[None, :]
    exact = mask.block_mask.mask_mod(torch.zeros(()), torch.zeros(()), q, k).numpy()
    assert np.array_equal(exact, expected)

    tiles = padded // TILE
    blocks = expected.reshape(tiles, TILE, tiles, TILE)
    any_tile, all_tile = blocks.any(axis=(1, 3)), blocks.all(axis=(1, 3))
    bm = mask.block_mask
    partial = np.zeros((tiles, tiles), dtype=bool)
    full = np.zeros((tiles, tiles), dtype=bool)
    for row in range(tiles):
        partial[row, bm.kv_indices[0, 0, row, : int(bm.kv_num_blocks[0, 0, row])].numpy()] = True
        full[
            row, bm.full_kv_indices[0, 0, row, : int(bm.full_kv_num_blocks[0, 0, row])].numpy()
        ] = True
    assert not (partial & full).any()
    assert np.array_equal(partial | full, any_tile)
    assert np.array_equal(full, all_tile)
    assert set(np.unique(mask.roles.numpy())) <= {
        PICTURE_ROLE,
        PROXY_ROLE,
        CLEAN_ROLE,
        NOISY_ROLE,
        PAD_ROLE,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels need CUDA")
def test_flex_attention_matches_masked_sdpa():
    doc = layout(target_latent_size=(8, 16), proxy_latent_size=(4, 8), frame_range=(0, 6))
    mask = build_window_mask(doc, window_chunks=2, windowed_clean=True, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            (1, doc.length, 2, 64), device="cuda", generator=generator, dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    out = mask.attend(q, k, v, layer=0).float()
    allowed = torch.from_numpy(dense_window_mask(doc, window_chunks=2, windowed_clean=True)).cuda()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        attn_mask=allowed,
    ).transpose(1, 2)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_rollout_cache_survives_fsdp_input_cast():
    from torch.distributed.utils import _apply_to_tensors

    from solarwm.backends.wan22.ref2va.attention import RolloutAttention, RolloutCache

    cache = RolloutCache(2)
    plan = RolloutAttention("prefill", cache, picture_tokens=1, proxy_chunk_spans=((1, 2),))
    cast = _apply_to_tensors(lambda t: t.to(torch.bfloat16), {"plan": plan})["plan"]
    key = torch.zeros(1, 2, 1, 4)
    cast.attend(key, key, key, layer=0)
    assert 0 in cache.pictures and 0 in cache.proxy
