# MiniMax-H3

MiniMax-H3 supports Stage0.5 FM, Stage1 TF-AnyFlow, and Stage2 SGF.
Training and fixed 158-frame inference use [preencoded data](../latent-wds.md).
Stage2 also supports full-length inference from a first image and camera trajectory.

## Setup

Activate the H3 environment from
[Runtime environments](../../environments/README.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

Accept the model repository's access terms on Hugging Face, then download the
base model if it is not already installed:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

For inference only, download `SolarWM-h3-33B-sgf-stage2-158f-fix`; Stage0.5 and
Stage1 checkpoints are not needed. The base model and the input dependencies
used by your inference command are still required.

```bash
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-sgf-stage2-158f-fix/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

The latent package supplies the files in `H3_SUPPORT`. Set the paths and
rendezvous address below; set `NODE_RANK` separately on each node.

```bash
export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_STAGE2_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-sgf-stage2-158f-fix"
export H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"
export NNODES=32
export NODE_RANK=0
export MASTER_ADDR=hostname-or-ip-of-node-0
export MASTER_PORT=29500
```

## Stage0.5 training

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f"
```

For a two-node run, set `NNODES=2` and add
`--set distributed.world_size=16 --set train.global_batch_size=8` to Stage0.5
or Stage1 training, or `--set train.global_batch_size=4` with the same world-size
override for Stage2. These smaller runs change the global batch size; the
unmodified configs retain the released recipes.

### Ref2VA proxy Stage0.5 training

**Ref2VA** means “reference-to-video-and-audio”: the transformer receives
ordered visual references before the target rows. The proxy profile reads the
existing FastVideo `.pt` cache directly; no WebDataset conversion is required.
Use the
[H3 proxy trainability checklist](../runbooks/h3-proxy-trainability-checklist.md)
when loss or visual evaluations appear not to move.
Its data contract is isolated from native `h3.158f.v1`:

- 124 pixel frames become 37 target latents;
- references are ordered as picture anchor, then video proxy;
- target, proxy and anchor keys are `vae_latent`, `proxy_latent` and
  `anchor_latent`;
- `text_embedding`, `text_token_tags` and `info.cwm_system` must match the
  cached CWM prompt role;
- the `w0` role fixes one leading target latent; `wn` fixes ten;
- camera conditioning and native camera validation are disabled.

**LoRA** (low-rank adaptation) trains small matrices attached to frozen model
layers. This profile starts fresh rank-128 LoRA matrices on the Q/K/V/output
attention projections of the 50 main blocks (200 target linear layers) and
loads the base model's `transformer_ref` partition.

The checked-in example targets the active W0/Qwen-2-FPS cache:

```bash
cd /workspace/SolarWM
export NNODES=2
export NODE_RANK=0  # set to 1 on the second node
export MASTER_ADDR=ip-or-hostname-of-node-0
export MASTER_PORT=29500

torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-124f-ref2va-proxy-sp1.yaml
```

`SP1` means sequence parallel size 1: each GPU processes a complete packed
sequence. On 16 GPUs with micro-batch 1 and accumulation 1, the logical
data-parallel and global batch sizes are both 16. The example keeps learning
rate `2e-5`; it does not silently scale the rate from an earlier global batch.

Edit or override these machine-specific paths before launch:

```bash
--set model.checkpoint_path=/data/models/MiniMax-H3
--set data.data_path=/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple
--set runtime.output_dir=/data/binghe/h3_proxy/solarwm-runs/gta-v2-w0-lora128
```

Keep `validation.validate_every_steps=0` and `validation.smoke_step=0` during
proxy training. Proxy visualization is a separate one-node inference job, so
it cannot stall or exhaust memory in the distributed training process.

The example enables rank-0 W&B scalar tracking. Install and authenticate once
in the image before launching:

```bash
python -m pip install 'wandb>=0.18'
wandb login
```

Only optimizer/checkpoint scalars are sent: loss, learning rate, gradient norm,
step time, peak allocated memory and checkpoint boundaries. `train/loss/*` is
the mean across the logical data-parallel group, while `train/loss_ema/*` is
its exponential moving average (EMA), controlled by
`runtime.tracking.loss_ema_beta` (default `0.95`). The EMA state starts again
when the training process restarts. Media evaluation is deliberately not
launched inside the distributed training process.
`validation.manual_steps: [48, 96, 144]` records the intended standalone
evaluation checkpoints; it does not start an evaluator. Run the external
evaluator against those completed checkpoint directories and publish its media
with the matching global step.

Rank 0 persists the W&B run identity at
`<runtime.output_dir>/wandb-run-id.txt`. A checkpoint resume using the same
output directory reconnects to that run with `resume: allow`; the other 15
ranks never initialize W&B. Set `runtime.tracking.entity` when the project
belongs to a team account. `runtime.tracking.log_media` must remain `false` for
this profile.

When the proxy dataset membership changes, do not use
`checkpoint.resume_from`: a full resume intentionally restores and verifies
the optimizer, scheduler, RNG, and exact reader path digest. Instead, warm-start
a new run from only the prior proxy LoRA weights:

```bash
--set checkpoint.initialization.student.path=/absolute/checkpoint_model_003000 \
--set checkpoint.initialization.student.weight_source=live \
--set checkpoint.initialization.student.stage=stage0p5
```

The new run starts at optimizer step zero with fresh optimizer, scheduler, and
EMA state; EMA is initialized from the loaded live adapter. The initialization
checkpoint identity is recorded in every new checkpoint contract. To resume the
new run later, keep its resolved initialization block and set
`checkpoint.resume_from` to one of the new run's complete checkpoints. Use a
full resume only when the data manifest and topology are unchanged and exact
continuation is intended.

### Ref2VA omni proxy (separate depth and semantic references)

The proxy profile's canvas and reference set come from the config instead of
being frozen at 768 x 1344 with one 192 x 336 DUV video. The time axis stays
fixed (124 frames, 37 latents, 207 audio latents). The "omni" example
`stage0p5-124f-ref2va-omni-704p-sp2.yaml` trains on a 1280 x 704 target with
three references, all at the target's resolution:

- `<Picture 1>`: the anchor, center-cropped to the target's framing and then
  scaled to short edge 2048 (3712 x 2048, a 128 x 232 latent);
- `<Video 1>`: a grey depth video (DUV's log-depth channel in all three
  channels), latent `[24, 37, 44, 80]`;
- `<Video 2>`: a flat-colour semantic video (the 12 CWM classes on a
  3 x 2 x 2 RGB lattice), latent `[24, 37, 44, 80]`.

The cache stores these as `proxy_latents` `[2, 24, 37, 44, 80]` with
`info.proxy_references: [depth, semantic]`, and it uses the hash-locked FastVideo
system role `w0_depth_semantic`, which names both videos. The reader requires
`data.proxy_references`, the cached references and the role to agree, and the
`model.latent_height/latent_width/rows_per_latent` fields to match the data
canvas (44, 80, 880). An absent `data.proxy_references` means the legacy
single `duv` video, whose encoder contract is unchanged.

One omni document is about 108k rows: 32,560 target rows, 2 x 32,560 proxy
rows, 7,424 anchor rows, plus the Qwen text. That is about 2.3 times a legacy
proxy document, so the example uses SP2. Proxy training accepts SP 1, 2, 4 or 8.
On 16 GPUs SP2 gives a global batch of 8, so the example runs 288 steps to see
roughly the same number of samples as 144 legacy steps at batch 16.

```bash
cd /workspace/SolarWM && git pull
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-124f-ref2va-omni-704p-sp2.yaml \
  --set data.data_path=/data/binghe/h3_proxy/cache/abot_720p_omni_704_qwen2 \
  --set runtime.output_dir=/data/binghe/h3_proxy/solarwm-runs/abot-720p-omni-lora128
```

A legacy proxy LoRA can warm-start an omni run through
`checkpoint.initialization`, because the LoRA targets are the same. A full
`checkpoint.resume_from` across the two profiles is refused, because the
encoder contract differs.

## Stage1 / Stage2 setup

For training across stages, download the three EMA packages from the
[SolarWM-H3 weight repository](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B).
Keep each directory intact under `SOLAR_MODEL_ROOT`.

```bash
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-bid-stage0p5-158f/**" \
            "SolarWM-h3-33B-tf-stage1-158f/**" \
            "SolarWM-h3-33B-sgf-stage2-158f-fix/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

| Package | EMA checkpoint |
|---|---|
| `SolarWM-h3-33B-bid-stage0p5-158f` | Stage0.5 step 10500 |
| `SolarWM-h3-33B-tf-stage1-158f` | Stage1 step 3000 |
| `SolarWM-h3-33B-sgf-stage2-158f-fix` | Stage2 step 3900 |

For Stage2 inference, we recommend
`SolarWM-h3-33B-sgf-stage2-158f-fix` (step 3900 EMA), which supersedes the
original step 1200 release and improves fine-detail quality, especially in tree
regions. Use the `-fix` checkpoint directory with
`checkpoint.weight_source=ema`.

Validation selects fixed cases from the test index automatically. Stage1 also
reads complete camera trajectories from the raw test data.

```bash
export H3_STAGE0P5_INIT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f"
export H3_STAGE1_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-tf-stage1-158f"
```

### Stage1 TF-AnyFlow

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage1-158f-lora384-w6-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage1-anyflow-158f"
```

### Stage2 SGF

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage2-158f-lora384-w6-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE1_CHECKPOINT" \
  --set checkpoint.initialization.teacher.path="$H3_STAGE0P5_INIT" \
  --set checkpoint.initialization.critic.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-158f"
```

#### SGF+ role split

`--set model.adapter.role_split=sgf_plus` trains
[SGF+](https://zihan-su.github.io/self-gradient-forcing-plus/): the student
keeps its shared LoRA for denoising and gains a second rank-384 `context`
LoRA on the 50 main blocks (300 linears, about 2.0B parameters) for context
writing. The context adapter serves the clean history rows of the gradient
replay and the committed chunk in each rollout KV-commit forward; noisy rows,
the image anchor, text and audio stay on the shared adapter. A Stage1 or SGF
initialization copies the shared adapter into both roles, so step 0 matches
SGF exactly. Teacher and critic are unchanged.

The default `shared` is plain SGF. SGF+ checkpoints record the
parameterization `peft-lora-r384-alpha384-sgf-plus`; resume and inference must
use the same `role_split`. Student logs add `student_grad_norm_denoise` and
`student_grad_norm_context` before clipping. LoRA tensors and their optimizer
states are replicated on every GPU, so the split adds roughly 30 GiB per GPU
(BF16 weights and gradients plus FP32 master and Adam moments).

## Resume training

Use the original resolved configuration and a complete training checkpoint:

```bash
export H3_PREVIOUS_RUN="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-158f"

torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config "$H3_PREVIOUS_RUN/resolved-config.json" \
  --set checkpoint.resume_from="$H3_PREVIOUS_RUN/checkpoint_model_000200" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-resumed"
```

## Inference

### Stage0.5

The released stage packages contain EMA weights.

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f" \
  --set checkpoint.weight_source=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f-infer"
```

### Ref2VA proxy cached-sample visualization

`infer-stage0p5-124f-ref2va-proxy-sp8.yaml` evaluates one checkpoint on one
8-GPU node. It selects deterministic samples from the existing pre-encoded
`.pt` cache, so all checkpoints use identical text, anchor, proxy, and target
tensors. This is a cached training-sample sanity evaluation, not a held-out
evaluation: this path does not re-encode raw validation JSON or media.

```bash
unset PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT

torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-proxy-sp8.yaml \
  --set checkpoint.resume_from=/data/binghe/h3_proxy/solarwm-runs/gta-v2-w0-lora128-3000-v3/checkpoint_model_000500 \
  --set checkpoint.weight_source=ema \
  --set runtime.output_dir=/data/binghe/h3_proxy/evals/solarwm-native/step-500 \
  --set runtime.tracking.run_name=step-500
```

Use `checkpoint_model_001000` and `checkpoint_model_003000` with matching
output directories for the other trained checkpoints. For the untrained base
model, pass `--set checkpoint.resume_from=null` and use a `step-0` output
directory.

Each sample directory contains `proxy.mp4`, `generated.mp4`, `target.mp4`, and
`compare.mp4`. The comparison panel is ordered
`proxy | prediction | target`. Successful completion publishes
`proxy-inference/COMPLETE.json`. Rank 0 also uploads each comparison video to
the configured W&B project; set `runtime.tracking.entity` if the project
belongs to a team account.

For a fixed-checkpoint proxy-use ablation, run the same cache, selection seed,
noise seed, and output geometry three times with
`validation.proxy_ablation=correct`, `shuffled`, and `static`. Shuffled mode
rotates in a different sample's full-rate proxy latent and its Qwen proxy
vision-token rows while retaining the original anchor, caption, target, and
noise. Static mode repeats the original proxy's first frame in both paths.
The rendered proxy column and inference manifest identify the condition that
was actually sampled.

The Qwen vision rows are the runs tagged `VIDEO_TAG` (0) in
`text_token_tags`; text is `TEXT_TAG` (1). Ablations before 2026-10-09 selected
the runs tagged 1, so their "shuffled" and "static" arms replaced caption and
label rows instead of the proxy's vision rows, while the proxy VAE latent was
replaced correctly. Re-run those arms before drawing conclusions from them.
With several proxy references, static mode freezes each video on its own first
frame and first Qwen block.

The omni example `infer-stage0p5-124f-ref2va-omni-704p-sp8.yaml` writes
`proxy_depth.mp4` and `proxy_semantic.mp4` instead of `proxy.mp4`. Its panel
order is `proxy_depth | proxy_semantic | prediction | target`.

### Stage2 SGF

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set checkpoint.weight_source=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-infer"
```

For a checkpoint without EMA weights, add `--set checkpoint.weight_source=live`.

### Full-length Stage2 inference

Generate each test video at its original length and frame rate using 8 GPUs
(SP8, NFE4, W6). Only the first image is encoded; the supplied camera trajectory
controls the generated video.

Download the [standalone test set](../data-access.md#standalone-test-set) first.
Create a one-case plan from its index; select more complete rows for a larger run:

```bash
export H3_TEST_PLAN="$SOLAR_OUTPUT_ROOT/h3-test-plan.json"
python - <<'PY'
import gzip
import json
import os
from pathlib import Path

index = Path(os.environ["SOLAR_TEST_ROOT"]) / "indexes/all.jsonl.gz"
with gzip.open(index, "rt") as handle:
    rows = [next(json.loads(line) for line in handle if line.strip())]
plan = Path(os.environ["H3_TEST_PLAN"])
plan.parent.mkdir(parents=True, exist_ok=True)
plan.write_text(json.dumps(rows, indent=2) + "\n")
PY
```

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-source-length-sp8.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set checkpoint.weight_source=ema \
  --set inference.expected_step=3900 \
  --set inference.plan="$H3_TEST_PLAN" \
  --set inference.dataset_root="$SOLAR_TEST_ROOT" \
  --set inference.work_dir="$SOLAR_OUTPUT_ROOT/h3-condition-cache" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-full-length"
```

`inference.plan` is a JSON list of selected raw test-index rows; the dataset root
can be a local path or `gs://` URI. Set `inference.expected_step` to the checkpoint
step and use `checkpoint.weight_source=live` for LIVE weights.

Keep every selected row intact. The reader uses these index fields:

| Fields | Meaning |
|---|---|
| `sample_id`, `key`, `dataset` | Sample identity; each `key` must be unique in the plan |
| `num_frames`, `fps`, `height`, `width`, `caption` | Source geometry, playback rate, and prompt |
| `shard`, `shard_size` | Archive path relative to the dataset root and its byte size |
| `video_member`, `camera_member`, `intrinsics_member`, `manifest_member` | Member paths within the archive |
| `shard_generation` | Also required for GCS inputs |

The camera member contains absolute `c2w` matrices, one per source frame;
intrinsics use the source image coordinates. Results include videos, comparison
previews and FPS measurements. For videos longer than 960 frames, add
`--set inference.stream_decode=true`.

## Optional preencoding

To prepare a new latent generation from [raw-WDS](../data-access.md):

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm preencode \
  --config configs/examples/minimax_h3/preencode-158f.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set preencode.output_root="$SOLAR_OUTPUT_ROOT/preencoded/minimax-h3-158f-768p-nomind-v1" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-preencode-158f"
```

## License

The MiniMax-H3 base model and released adapters remain subject to the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
included with the model packages. Review its terms, including territorial
restrictions, before download, use, or redistribution. SolarWM's code license
does not replace it.
