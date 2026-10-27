import pytest
import torch

from solarwm.backends.wan22.ref2va.data import SCHEMA, Ref2VAWanSpec, Ref2VAWanStream, read_sample


def config(path, family="wan22_ti2v_5b", **data):
    return {
        "model": {"family": family},
        "data": {
            "data_path": str(path),
            "encoded_latents": 3,
            "latent_height": 4,
            "latent_width": 4,
            "proxy_latent_height": 2,
            "proxy_latent_width": 2,
            "oc_reference": True,
            "oc_drop_prob": 0.5,
            **data,
        },
    }


def write_sample(path, *, family="wan22_ti2v_5b", oc=True):
    c = 48 if family == "wan22_ti2v_5b" else 16
    payload = {
        "schema": SCHEMA,
        "family": family,
        "target_latent": torch.randn(c, 3, 4, 4).bfloat16(),
        "proxy_latent": torch.randn(c, 3, 2, 2).bfloat16(),
        "anchor_latent": torch.randn(c, 1, 6, 8).bfloat16(),
        "prompt_embeds": torch.randn(9, 4096).bfloat16(),
        "info": {"name": path.stem},
    }
    if oc:
        payload["oc_latent"] = torch.randn(c, 1, 4, 4).bfloat16()
        payload["prompt_embeds_no_oc"] = torch.randn(5, 4096).bfloat16()
    if family == "wan22_i2v_a14b":
        payload["i2v_latent"] = torch.randn(16, 3, 4, 4).bfloat16()
    torch.save(payload, path)
    return path


def test_read_sample_oc_rules(tmp_path):
    path = write_sample(tmp_path / "a.pt")
    spec = Ref2VAWanSpec.from_config(config(tmp_path))
    with_oc = read_sample(path, spec, use_oc=True)
    assert (
        len(with_oc.pictures) == 2 and with_oc.prompt.shape == (9, 4096) and not with_oc.oc_dropped
    )
    dropped = read_sample(path, spec, use_oc=True, drop_oc=True)
    assert len(dropped.pictures) == 1 and dropped.prompt.shape == (5, 4096) and dropped.oc_dropped
    no_oc_run = read_sample(path, spec, use_oc=False)
    assert (
        len(no_oc_run.pictures) == 1
        and no_oc_run.prompt.shape == (5, 4096)
        and not no_oc_run.oc_dropped
    )
    plain = read_sample(write_sample(tmp_path / "b.pt", oc=False), spec, use_oc=True)
    assert len(plain.pictures) == 1 and plain.prompt.shape == (9, 4096)


def test_read_sample_checks_family_and_i2v(tmp_path):
    path = write_sample(tmp_path / "a.pt")
    with pytest.raises(ValueError, match="encoded for"):
        read_sample(
            path, Ref2VAWanSpec.from_config(config(tmp_path, family="wan22_i2v_a14b")), use_oc=True
        )
    a14b = write_sample(tmp_path / "c.pt", family="wan22_i2v_a14b")
    sample = read_sample(
        a14b, Ref2VAWanSpec.from_config(config(tmp_path, family="wan22_i2v_a14b")), use_oc=True
    )
    assert sample.i2v_latent.shape == (16, 3, 4, 4)


def test_stream_ranks_disjoint_and_resumable(tmp_path):
    for index in range(5):
        write_sample(tmp_path / f"s{index}.pt", oc=index % 2 == 0)
    streams = [Ref2VAWanStream(config(tmp_path), rank=rank, world_size=2) for rank in range(2)]
    first_epoch = [[s.next().sample_id for _ in range(3)] for s in streams]
    assert set(first_epoch[0]) | set(first_epoch[1]) == {f"s{i}" for i in range(5)}

    reference = Ref2VAWanStream(config(tmp_path), rank=0, world_size=2)
    for _ in range(4):
        reference.next()
    state = reference.state_dict()
    expected = [(item.sample_id, item.oc_dropped) for item in (reference.next() for _ in range(4))]
    resumed = Ref2VAWanStream(config(tmp_path), rank=0, world_size=2)
    resumed.load_state_dict(state)
    assert [
        (item.sample_id, item.oc_dropped) for item in (resumed.next() for _ in range(4))
    ] == expected

    eval_stream = Ref2VAWanStream(config(tmp_path), rank=0, world_size=1, training=False)
    assert not any(eval_stream.next().oc_dropped for _ in range(10))


@pytest.mark.parametrize("new_world", [4, 16])
def test_stream_resumes_at_another_world_size(tmp_path, new_world):
    for index in range(64):
        write_sample(tmp_path / f"s{index}.pt", oc=False)
    old = [Ref2VAWanStream(config(tmp_path), rank=rank, world_size=8) for rank in range(8)]
    consumed = [s.next().sample_id for _ in range(4) for s in old]
    state = old[0].state_dict()
    expected = [s.next().sample_id for _ in range(2) for s in old]

    new = [
        Ref2VAWanStream(config(tmp_path), rank=rank, world_size=new_world)
        for rank in range(new_world)
    ]
    for stream in new:
        stream.load_state_dict(state)
    following = [s.next().sample_id for _ in range(16 // new_world) for s in new]
    assert sorted(following) == sorted(expected)
    assert not set(following) & set(consumed)
