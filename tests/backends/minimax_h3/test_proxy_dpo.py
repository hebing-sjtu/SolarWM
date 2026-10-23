from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from solarwm.backends.minimax_h3.runtime import H3TrainingRuntime  # noqa: E402
from solarwm.errors import DataContractError  # noqa: E402


class _Core:
    """Flow-matching error of a linear 'model' whose weight is the LoRA parameter."""

    def __init__(self, weight: torch.nn.Parameter) -> None:
        self.weight = weight

    def forward_loss(self, batch, *, noise_seed, target=None):
        generator = torch.Generator().manual_seed(int(noise_seed))
        noise = torch.randn(target.shape, generator=generator)
        return ((self.weight * noise - target) ** 2).mean()


def _runtime(weight: torch.nn.Parameter, reference: torch.Tensor, beta: float, sft: float):
    runtime = SimpleNamespace(
        torch=torch,
        device="cpu",
        train_cfg={"dpo": {"beta": beta, "sft_weight": sft}},
        core=_Core(weight),
        lora=SimpleNamespace(parameters=(weight,)),
        _reference_lora=[reference],
        dist=SimpleNamespace(is_initialized=lambda: False),
    )
    runtime._reference_weights = H3TrainingRuntime._reference_weights.__get__(runtime)
    return runtime


@pytest.mark.parametrize("beta,sft", [(50.0, 0.0), (500.0, 0.3)])
def test_dpo_microbatch_gradient_matches_autograd_of_the_full_loss(beta: float, sft: float) -> None:
    torch.manual_seed(0)
    win, lose = torch.randn(4, 3), torch.randn(4, 3)
    batch = SimpleNamespace(sample_id="s", target_latents=win, target_latents_lose=lose)
    reference = torch.full((4, 3), 0.7)
    weight = torch.nn.Parameter(torch.full((4, 3), 0.9))

    torch.manual_seed(123)
    stats = H3TrainingRuntime._dpo_microbatch(_runtime(weight, reference, beta, sft), batch, 2.0)
    split = weight.grad.clone()
    assert torch.equal(weight.detach(), torch.full((4, 3), 0.9)), "reference swap must restore the live weight"

    torch.manual_seed(123)
    seed = int(torch.randint(0, 2**31 - 1, ()).item())
    core = _Core(weight)
    with torch.no_grad():
        core.weight = reference
        ref_win = core.forward_loss(batch, noise_seed=seed, target=win)
        ref_lose = core.forward_loss(batch, noise_seed=seed, target=lose)
    core.weight = weight
    weight.grad = None
    pol_win = core.forward_loss(batch, noise_seed=seed, target=win)
    pol_lose = core.forward_loss(batch, noise_seed=seed, target=lose)
    margin = (pol_win - ref_win) - (pol_lose - ref_lose)
    loss = -torch.nn.functional.logsigmoid(-beta * margin) + sft * pol_win
    (loss / 2.0).backward()

    torch.testing.assert_close(split, weight.grad, rtol=1e-5, atol=1e-7)
    assert stats["dpo"] == pytest.approx(float(loss - sft * pol_win), rel=1e-5)
    assert stats["dpo_accuracy"] == float(margin < 0)
    assert stats["dpo_reward_margin"] == pytest.approx(float(-margin), rel=1e-5, abs=1e-8)


def test_dpo_microbatch_refuses_a_sample_without_a_rejected_latent() -> None:
    weight = torch.nn.Parameter(torch.ones(2))
    batch = SimpleNamespace(sample_id="s", target_latents=torch.ones(2), target_latents_lose=None)
    with pytest.raises(DataContractError, match="vae_latent_lose"):
        H3TrainingRuntime._dpo_microbatch(_runtime(weight, torch.ones(2), 1.0, 0.0), batch, 1.0)
