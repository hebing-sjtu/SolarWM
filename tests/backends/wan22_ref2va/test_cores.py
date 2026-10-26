from dataclasses import dataclass

import pytest
import torch

from solarwm.backends.wan22.ref2va import cores as cores_module
from solarwm.backends.wan22.ref2va.data import Ref2VAWanSample
from solarwm.backends.wan22.ref2va.layout import NOISY_ROLE, dense_window_mask
from solarwm.backends.wan22.ref2va.model import Ref2VAWanModel

TINY = dict(dim=48, ffn_dim=64, num_heads=4, num_layers=2)
FAMILIES = {"wan22_ti2v_5b": 48, "wan22_i2v_a14b": 16}


@dataclass
class DensePlan:
    allowed: torch.Tensor

    def attend(self, q, k, v, *, layer):
        del layer
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=self.allowed
        )
        return out.transpose(1, 2)


def dense_mask(self, layout, *, windowed_clean):
    allowed = dense_window_mask(
        layout, window_chunks=self.spec.window_chunks, windowed_clean=windowed_clean
    )
    return DensePlan(torch.from_numpy(allowed))


def config(family, *, given=1, latents=5):
    return {
        "model": {"family": family, "reference_i2v_y": "self"},
        "data": {
            "encoded_latents": latents,
            "latent_height": 4,
            "latent_width": 4,
            "proxy_latent_height": 2,
            "proxy_latent_width": 2,
            "num_given_latent_frames": given,
            "chunk_latents": 2,
            "window_chunks": 2,
            "train_target_latents": latents,
        },
        "train": {
            "timestep_shift": 5.0,
            "diffusion_ratio": 0.5,
            "consistency_ratio": 0.25,
            "finite_difference_epsilon": 5.0,
            "num_denoising_steps": 4,
            "score_min_sigma": 0.02,
            "score_max_sigma": 0.98,
        },
    }


def tiny_model(family, *, anyflow_gate=None, seed=0):
    torch.manual_seed(seed)
    model = Ref2VAWanModel(family, anyflow_gate=anyflow_gate, **TINY).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model


def sample(family, *, latents=5, oc=True, seed=0):
    c = FAMILIES[family]
    g = torch.Generator().manual_seed(seed)
    pictures = (torch.randn((c, 1, 4, 6), generator=g),)
    if oc:
        pictures += (torch.randn((c, 1, 2, 4), generator=g),)
    return Ref2VAWanSample(
        sample_id="s",
        target=torch.randn((c, latents, 4, 4), generator=g),
        proxy=torch.randn((c, latents, 2, 2), generator=g),
        pictures=pictures,
        prompt=torch.randn((6, 4096), generator=g),
        i2v_latent=torch.randn((16, latents, 4, 4), generator=g)
        if family == "wan22_i2v_a14b"
        else None,
        info={},
        fingerprint="test",
    )


@pytest.fixture(autouse=True)
def dense_plans(monkeypatch):
    monkeypatch.setattr(cores_module.Ref2VAWanBDCore, "mask", dense_mask)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("exit_index", [0, 3])
def test_sgf_cached_rollout_matches_windowed_replay(family, exit_index):
    model = tiny_model(family, anyflow_gate=0.25)
    core = cores_module.Ref2VAWanSGFCore(model, "cpu", config(family))
    inputs = core.prepare(sample(family), core.generator(7))
    rollout = core.rollout(inputs, generator=core.generator(8), exit_index=exit_index)
    assert rollout.cache_target.shape[2] == core.spec.rollout_latents == 6
    with torch.no_grad():
        replayed = core.replay(inputs, rollout)
    given = inputs.given
    torch.testing.assert_close(
        replayed[:, :, given:], rollout.cache_target[:, :, given:], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(rollout.cache_target[:, :, :given], inputs.given_value)


def test_sgf_replay_carries_gradients_and_scores():
    family = "wan22_ti2v_5b"
    student = tiny_model(family, anyflow_gate=0.25)
    teacher = tiny_model(family, seed=1)
    core = cores_module.Ref2VAWanSGFCore(student, "cpu", config(family))
    inputs = core.prepare(sample(family), core.generator(1))
    rollout = core.rollout(inputs, generator=core.generator(2))
    video = core.supervised(core.replay(inputs, rollout))
    assert video.shape[2] == core.spec.latent_frames
    noisy, noise, sigma = core.score_inputs(video.detach(), inputs, generator=core.generator(3))
    assert 0.02 <= float(sigma) <= 0.98
    with torch.no_grad():
        velocity = core.score_forward(teacher, inputs, noisy, sigma)
    assert velocity.shape == video.shape
    (video * velocity).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert core.kl_mask(inputs)[:, :, : inputs.given].sum() == 0
    assert torch.isfinite(
        core.critic_loss(velocity, noise=noise, clean=video.detach(), inputs=inputs)
    )


def test_sgf_requires_all_cached_latents():
    cfg = config("wan22_ti2v_5b")
    cfg["data"]["train_target_latents"] = 4
    with pytest.raises(ValueError, match="train_target_latents"):
        cores_module.Ref2VAWanSGFCore(tiny_model("wan22_ti2v_5b"), "cpu", cfg)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_bd_loss_and_generate(family):
    model = tiny_model(family)
    core = cores_module.Ref2VAWanBDCore(model, "cpu", config(family))
    loss = core.forward_loss(sample(family), noise_seed=3)
    loss.backward()
    assert torch.isfinite(loss) and model.patch_embedding.weight.grad is not None
    video = core.generate(sample(family), noise_seed=4, num_inference_steps=4)
    assert video.shape == (1, FAMILIES[family], 5, 4, 4)
    torch.testing.assert_close(
        video[:, :, :1], core.prepare(sample(family), core.generator(4)).given_value
    )


def test_bd_given_latents_carry_no_signal():
    family = "wan22_ti2v_5b"
    model = tiny_model(family)
    core = cores_module.Ref2VAWanBDCore(model, "cpu", config(family))
    item = sample(family)
    inputs = core.prepare(item, core.generator(0))
    layout, _ = core.layout(inputs, frame_range=(0, 5), teacher_forcing=False)
    times = core.token_times(layout, condition=1.0, noisy=500.0, given=1, given_time=1.0)
    noisy = layout.only(NOISY_ROLE)
    assert times[noisy.start : noisy.start + noisy.tokens_per_frame].eq(1.0).all()
    assert times[noisy.start + noisy.tokens_per_frame :].eq(500.0).all()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_ar_loss_and_rollout(family):
    model = tiny_model(family, anyflow_gate=0.25)
    cfg = config(family, latents=6)
    core = cores_module.Ref2VAWanARCore(model, "cpu", cfg)
    loss = core.forward_loss(sample(family, latents=6), noise_seed=5)
    loss.backward()
    assert torch.isfinite(loss) and model.delta_embedding[0].weight.grad is not None
    video = core.generate(sample(family, latents=6), noise_seed=6, num_inference_steps=4)
    assert video.shape == (1, FAMILIES[family], 6, 4, 4)


def test_ar_noisy_first_drops_only_unread_rows():
    """Dropping earlier chunks' noisy copies (``noisy_first``) leaves the current chunk as is."""

    family = "wan22_ti2v_5b"
    model = tiny_model(family, anyflow_gate=0.25)
    core = cores_module.Ref2VAWanARCore(model, "cpu", config(family, latents=6, given=0))
    inputs = core.prepare(sample(family, latents=6), core.generator(0))
    g = torch.Generator().manual_seed(1)
    clean = torch.randn((1, 48, 4, 4, 4), generator=g)
    noisy = torch.randn((1, 48, 4, 4, 4), generator=g)
    with torch.no_grad():
        full, coords = core.layout(inputs, frame_range=(2, 6), teacher_forcing=True)
        t = core.token_times(full, condition=1.0, noisy=600.0)
        reference = core.run(
            model,
            full,
            coords,
            core.segment_tensors(full, inputs, clean=clean, noisy=noisy),
            inputs,
            t,
            r_timesteps=t,
            plan=core.mask(full, windowed_clean=False),
        )
        window, wcoords = core.layout(
            inputs, frame_range=(2, 6), teacher_forcing=True, noisy_first=4
        )
        wt = core.token_times(window, condition=1.0, noisy=600.0)
        local = core.run(
            model,
            window,
            wcoords,
            core.segment_tensors(window, inputs, clean=clean[:, :, :2], noisy=noisy[:, :, 2:]),
            inputs,
            wt,
            r_timesteps=wt,
            plan=core.mask(window, windowed_clean=False),
        )
    torch.testing.assert_close(local, reference[:, :, 2:], atol=1e-4, rtol=1e-4)
