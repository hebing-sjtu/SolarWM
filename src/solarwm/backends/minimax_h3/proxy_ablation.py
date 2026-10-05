"""Condition-only ablations for cached MiniMax-H3 Ref2VA proxy samples."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from solarwm.errors import BackendContractError

from .proxy_artifacts import H3ProxyArtifactBatch

PROXY_ABLATION_MODES = frozenset({"correct", "shuffled", "static"})


def _vision_runs(tags: Any) -> tuple[slice, ...]:
    """Return contiguous Qwen vision-token runs; picture is first, video follows."""

    values = tags.detach().cpu().tolist()
    runs: list[slice] = []
    start: int | None = None
    for index, value in enumerate((*values, 0)):
        if int(value) == 1 and start is None:
            start = index
        elif int(value) != 1 and start is not None:
            runs.append(slice(start, index))
            start = None
    if len(runs) < 2:
        raise BackendContractError(
            "H3 proxy prompt must contain one picture vision run and at least one video run"
        )
    return tuple(runs)


def _replace_proxy_prompt_rows(
    prompt: Any,
    tags: Any,
    *,
    source_prompt: Any,
    source_tags: Any,
    static: bool,
) -> Any:
    destination_runs = _vision_runs(tags)[1:]
    source_runs = _vision_runs(source_tags)[1:]
    if len(destination_runs) != len(source_runs):
        raise BackendContractError("H3 proxy Qwen video block counts differ during ablation")

    def fit_rows(source: slice, count: int) -> Any:
        rows = source_prompt[source]
        source_count = int(rows.shape[0])
        if source_count == count:
            return rows
        if count == 1:
            return rows[:1]
        denominator = count - 1
        indices = [
            (index * (source_count - 1) + denominator // 2) // denominator
            for index in range(count)
        ]
        return rows[indices]

    output = prompt.clone()
    first_source = source_runs[0]
    for index, destination in enumerate(destination_runs):
        source = first_source if static else source_runs[index]
        destination_count = int(destination.stop - destination.start)
        output[destination] = fit_rows(source, destination_count).to(
            device=output.device,
            dtype=output.dtype,
        )
    return output


def apply_proxy_ablation(
    batch: H3ProxyArtifactBatch,
    *,
    mode: str,
    donor: H3ProxyArtifactBatch | None = None,
) -> H3ProxyArtifactBatch:
    """Replace only proxy conditioning while preserving anchor, caption, and target."""

    selected = str(mode).strip().lower()
    if selected not in PROXY_ABLATION_MODES:
        raise BackendContractError(f"unsupported H3 proxy ablation mode {mode!r}")
    if selected == "correct":
        if donor is not None:
            raise BackendContractError("correct proxy ablation must not receive a donor")
        return batch

    if selected == "shuffled":
        if donor is None or donor.sample_id == batch.sample_id:
            raise BackendContractError("shuffled proxy ablation requires a different donor sample")
        if (
            donor.cwm_system != batch.cwm_system
            or donor.num_given_latent_frames != batch.num_given_latent_frames
        ):
            raise BackendContractError("shuffled proxy donor has a different CWM contract")
        proxy = donor.proxy_latents.clone()
        prompt = _replace_proxy_prompt_rows(
            batch.prompt_embeds,
            batch.text_token_tags,
            source_prompt=donor.prompt_embeds,
            source_tags=donor.text_token_tags,
            static=False,
        )
        donor_identity = donor.plan_fingerprint
    else:
        proxy = batch.proxy_latents[:, :1].expand_as(batch.proxy_latents).clone()
        prompt = _replace_proxy_prompt_rows(
            batch.prompt_embeds,
            batch.text_token_tags,
            source_prompt=batch.prompt_embeds,
            source_tags=batch.text_token_tags,
            static=True,
        )
        donor_identity = "first-proxy-frame"

    fingerprint = hashlib.blake2s(
        (f"{batch.plan_fingerprint}\0proxy-ablation={selected}\0donor={donor_identity}").encode()
    ).hexdigest()
    return replace(
        batch,
        plan_fingerprint=fingerprint,
        proxy_latents=proxy,
        prompt_embeds=prompt,
    )


__all__ = ["PROXY_ABLATION_MODES", "apply_proxy_ablation"]
