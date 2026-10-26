"""Single-GPU smoke of the whole Wan Ref2VA chain on a tiny random model.

Stage0.5 -> Stage1 -> Stage2 hand off checkpoints exactly as the example configs do, Stage2
resumes, and the LoRA path saves adapters only. Run on a pod with ``pytest -m '' -k gpu``.
"""

import copy
import json

import pytest
import torch

from solarwm.backends.wan22.ref2va.data import SCHEMA

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")

TINY = {"dim": 64, "ffn_dim": 96, "num_heads": 4, "num_layers": 2}


def write_samples(directory, count=3):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        payload = {
            "schema": SCHEMA,
            "family": "wan22_ti2v_5b",
            "target_latent": torch.randn(48, 6, 4, 4).bfloat16(),
            "proxy_latent": torch.randn(48, 6, 2, 2).bfloat16(),
            "anchor_latent": torch.randn(48, 1, 6, 8).bfloat16(),
            "prompt_embeds": torch.randn(9, 4096).bfloat16(),
            "info": {"name": f"s{index}"},
        }
        if index % 2 == 0:
            payload["oc_latent"] = torch.randn(48, 1, 4, 4).bfloat16()
            payload["prompt_embeds_no_oc"] = torch.randn(5, 4096).bfloat16()
        torch.save(payload, directory / f"s{index}.pt")
    return directory


def base_config(tmp_path, stage, name, *, mode="full"):
    optimizer = {
        "learning_rate": 1e-4,
        "betas": [0.0, 0.999],
        "gradient_clip": 10.0,
        "warmup_steps": 0,
    }
    config = {
        "model": {
            "family": "wan22_ti2v_5b",
            "conditioning_mode": "ref2va_proxy",
            "weights": None,
            "architecture_overrides": TINY,
            "training_mode": mode,
        },
        "data": {
            "data_path": str(tmp_path / "samples"),
            "encoded_latents": 6,
            "latent_height": 4,
            "latent_width": 4,
            "proxy_latent_height": 2,
            "proxy_latent_width": 2,
            "num_given_latent_frames": 1,
            "oc_reference": True,
            "oc_drop_prob": 0.2,
            "chunk_latents": 2,
            "window_chunks": 2,
            "train_target_latents": 6,
            "seed": 3,
        },
        "train": {
            "stage": stage,
            "max_steps": 2,
            "global_batch_size": 1,
            "timestep_shift": 5.0,
            "anyflow_gate": 0.25,
            "diffusion_ratio": 0.5,
            "consistency_ratio": 0.25,
            "finite_difference_epsilon": 5.0,
            "num_denoising_steps": 4,
            "score_min_sigma": 0.02,
            "score_max_sigma": 0.98,
            "critic_updates_per_student": 1,
            "optimizer": optimizer,
            "critic_optimizer": dict(optimizer, learning_rate=2e-5),
            "fsdp": {"activation_checkpointing": True},
        },
        "checkpoint": {
            "save_every_steps": 1,
            "weights_dtype": "float32",
            "ema": {"enabled": True, "decay": 0.9, "start_step": 1},
        },
        "runtime": {"output_dir": str(tmp_path / name), "tracking": {"enabled": False}},
    }
    if mode == "lora":
        config["model"]["adapter"] = {
            "target": "main_attention_qkvo",
            "rank": 4,
            "alpha": 4,
            "expected_target_linear_modules": 8,
        }
    return config


def train(config):
    from solarwm.backends.wan22.ref2va.runtime import Ref2VAWanTrainer

    trainer = Ref2VAWanTrainer(config)
    assert trainer.run() == 0
    return trainer


def events(directory):
    return [
        json.loads(line) for line in (directory / "training-events.jsonl").read_text().splitlines()
    ]


def test_gpu_full_chain_and_resume(tmp_path):
    write_samples(tmp_path / "samples")
    bd = base_config(tmp_path, "stage0p5", "bd")
    train(bd)
    bd_checkpoint = tmp_path / "bd" / "checkpoint-000002"
    assert (bd_checkpoint / "COMPLETE.json").is_file() and (
        bd_checkpoint / "student-ema.safetensors"
    ).is_file()

    ar = base_config(tmp_path, "stage1", "ar")
    ar["checkpoint"]["initialization"] = {
        "student": {"path": str(bd_checkpoint), "role": "student"}
    }
    trainer = train(ar)
    assert trainer.student.identity["init"]["delta"] is False
    ar_checkpoint = tmp_path / "ar" / "checkpoint-000002"
    assert all(
        e["loss"] == e["loss"] for e in events(tmp_path / "ar") if e["event"] == "train_step"
    )

    sgf = base_config(tmp_path, "stage2", "sgf")
    sgf["checkpoint"]["initialization"] = {
        "student": {"path": str(ar_checkpoint), "role": "student"},
        "teacher": {"path": str(bd_checkpoint), "role": "student"},
        "critic": {"path": str(bd_checkpoint), "role": "student"},
    }
    trainer = train(sgf)
    assert trainer.student.identity["init"]["delta"] is True
    steps = [e for e in events(tmp_path / "sgf") if e["event"] == "train_step"]
    assert [e["student_updated"] for e in steps] == [0.0, 1.0]

    from safetensors.torch import load_file

    assert not (tmp_path / "sgf" / "checkpoint-000001" / "student-resume-rank000.pt").exists()
    assert (tmp_path / "sgf" / "checkpoint-000002" / "critic-resume-rank000.pt").is_file()
    resumed = copy.deepcopy(sgf)
    resumed["train"]["max_steps"] = 3
    resumed["checkpoint"]["resume_from"] = str(tmp_path / "sgf" / "checkpoint-000002")
    trainer = train(resumed)
    assert trainer.global_step == 3

    straight = copy.deepcopy(sgf)
    straight["train"]["max_steps"] = 3
    straight["runtime"]["output_dir"] = str(tmp_path / "sgf3")
    train(straight)
    a = load_file(str(tmp_path / "sgf" / "checkpoint-000003" / "student.safetensors"))
    b = load_file(str(tmp_path / "sgf3" / "checkpoint-000003" / "student.safetensors"))
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key].float(), b[key].float(), atol=2e-2, rtol=2e-2)


def test_gpu_lora_saves_adapters_only(tmp_path):
    from safetensors.torch import load_file

    write_samples(tmp_path / "samples")
    train(base_config(tmp_path, "stage1", "lora", mode="lora"))
    state = load_file(str(tmp_path / "lora" / "checkpoint-000002" / "student.safetensors"))
    assert state and all(".lora_" in key or key.startswith("delta_embedding.") for key in state)
    assert sum(".lora_" in key for key in state) == 16
