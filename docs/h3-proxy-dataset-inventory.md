# H3 proxy experiment dataset inventory

Last updated: 2026-10-03

This is the working inventory for the GTA/ABot proxy-to-video experiments that
were prepared in FastVideo, plus the native MiniMax-H3 data route supported by
SolarWM. It records the semantic contract as well as the path: two directories
with the same tensor shapes are not interchangeable when their prompt,
reference order, Qwen sampling rate, or DUV convention differs.

The `/data/binghe/...` paths are cluster-local experiment assets. They are not
part of the public SolarWM-Data release and must not be published without a
separate data and license review.

## Short decision guide

- Use `gta_v2_cwm_1344_qwen2_simple` for the current FastVideo simple-prompt,
  W0, Qwen-2-FPS, no-time-alignment experiment.
- Use `gta_v2_cwm_1344_qwen2_detailed` for the matched detailed-prompt arm.
  Its visual latents are copied from the same source cache; the intended
  experimental variable is the cached prompt embedding.
- Use `gta_v2_cwm_1344_qwen24` and
  `gta_v2_cwm_1344_qwen24_wn` only for the 24-FPS CWM W0/WN route.
- Use `abot_moge3_sam2_w0_qwen2` for the strictly filtered ABot experiment:
  metric MoGe3 depth, SAM2-refined CWM12 semantics, W0, and Qwen 2 FPS.
- Use SolarWM's published
  `minimax-h3-158f-768p-nomind-v1` latent-WDS for an experiment that should
  run on the current SolarWM MiniMax-H3 backend without backend changes.
- FastVideo proxy `.pt` caches are accepted only through SolarWM's isolated
  `proxy_preencoded` Stage0.5 profile. They are not native `h3.158f.v1` data
  and cannot be used for native camera-conditioned Stage1/Stage2.

## GTA proxy corpus

### Raw source

Root:

```text
/data/binghe/datasets/gta_web_0902_v2/gta_web_0902
```

The relevant nested layout is:

```text
seg_NNNN/
  prompt.json
  prompt.txt                         # preferred when present
  metadata.json
  minimax_h3/output.mp4             # photoreal target
  minimax_h3/image_1.png            # optional style image; not the default anchor
  proxy/duv.mp4                     # packed DUV proxy
```

The active cache contract is:

- target: 124 frames on a 24-FPS timeline;
- encoded target canvas: 768 x 1344;
- DUV proxy canvas: 192 x 336;
- anchor: target frame zero, resized to short edge 2048;
- reference order: `<Picture 1>` followed by `<Video 1>`;
- W0: one given latent frame, hard-locked from the anchor;
- proxy VAE input: all 124 frames, independently of Qwen's preview FPS.

`minimax_h3/image_1.png` is a style reference and is not guaranteed to equal
target frame zero. The current W0 hard-lock route therefore uses
`anchor_source=target`; switching to `image1` changes the first frame and is
not a prompt-only experiment.

### Corpus membership

Primary encode manifest:

```text
/data/binghe/h3_proxy/gta_v2_cwm_encode.jsonl
```

Observed inventory:

- 771 filtered manifest rows;
- 762 successfully cached clips;
- nine historical manifest rows did not produce usable cache samples.

The cache directory, rather than the unfiltered manifest row count, is the
authoritative training membership. The Qwen-2-FPS manifests were filtered
against the existing 768 x 1344 cache so their rows correspond to actual
`.pt` files.

### Training manifests and prompt assets

Simple-prompt training manifest:

```text
/data/binghe/h3_proxy/gta_v2_simple_qwen2_train.jsonl
```

Detailed/colleague-style training manifest:

```text
/data/binghe/h3_proxy/gta_v2_detailed_qwen2_train.jsonl
```

Assets used to generate the detailed captions:

```text
/data/binghe/h3_proxy/gta_v2_train_caption_input.json
/data/binghe/h3_proxy/low_high_train_colleague/high_motion/
/data/binghe/h3_proxy/colleague_captions_train.json
```

The detailed captions were generated from the matching photoreal target clips
with Vertex Gemini, then adapted to the one-anchor plus one-proxy H3 contract.
They use natural visual proxy wording and explicit motion/camera narration.
The simple captions are the original CWM-shaped user captions.

Both arms still use the same packaged CWM W0 system prompt. “Simple versus
detailed” refers to the user caption, not to removing or changing the system
prompt.

### Validation sets

Simple validation:

```text
/data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json
```

Detailed validation:

```text
/data/binghe/h3_proxy/gta_v2_colleague_validation_val6.json
```

Both contain the same six clip IDs and the same media paths. The intended
difference is only `data[].caption`. They should always be evaluated with the
same checkpoint, seed, sampling steps, guidance, reference order, first-frame
contract, alignment setting, and Qwen video FPS.

The validation JSON stores raw media paths. It does not contain cached text
embeddings; evaluation re-encodes the selected caption and visual references.

### Active FastVideo caches

#### W0, simple caption, Qwen 24 FPS

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24
```

Contract:

- 768 x 1344 target;
- 192 x 336 proxy;
- anchor short edge 2048;
- 124 target/proxy frames;
- CWM system role `w0`;
- simple caption;
- Qwen sees the proxy at 24 FPS.

#### WN, simple caption, Qwen 24 FPS

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24_wn
```

The visual latents match the W0 cache. The text embedding uses the CWM `wn`
system contract, paired with ten given latent frames during training and
sampling.

#### W0, simple caption, Qwen 2 FPS

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple
```

This is the active simple-prompt arm. It retains the current 768 x 1344 visual
latents and rewrites the text-side conditioning with:

- CWM `w0`;
- simple user caption;
- Qwen proxy preview at 2 FPS.

The current comparison trains and evaluates it with proxy time alignment
disabled.

#### W0, detailed caption, Qwen 2 FPS

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_detailed
```

This is the active detailed-prompt arm. It is a text-only rewrite of the same
visual cache using `gta_v2_detailed_qwen2_train.jsonl`. It must have exactly
the same `.pt` filenames as the simple arm.

### Important cache semantics

Time alignment is not encoded into a cache. It is a packing-time model setting:

```text
models.student.align_proxy_reference_time
callbacks.validation.align_proxy_reference_time
```

The two values must match. The active Qwen-2-FPS simple/detailed experiment
sets both to `false`.

Qwen video FPS affects only the visual frames presented to Qwen while building
the text/prompt embedding. The proxy VAE always encodes the complete 24-FPS
proxy clip. Therefore a Qwen-2-FPS cache and Qwen-24-FPS cache can have
identical target/proxy latents but are not the same training input.

Reference order is part of the token contract. These caches were encoded as:

```text
<Picture 1> anchor -> <Video 1> proxy
```

Do not evaluate a trained checkpoint with `video_picture`.

## Historical and secondary FastVideo assets

These paths are useful for provenance or plumbing tests but should not replace
an active cache silently.

### Legacy GTA CWM cache

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm
/data/binghe/h3_proxy/cache/gta_v2_cwm_wn
```

Known geometry is 704 x 1280 target over a 192 x 336 proxy. These caches
predate the current released 768 x 1344 geometry. Historical runs and older
Qwen-2-FPS comparisons may refer to them.

### Older GTA training cache

```text
/data/binghe/h3_proxy/cache/gta_v2_train
```

Known geometry is 768 x 1344 over 192 x 336, but its DUV/prompt provenance
predates the current CWM cache contract. It is suitable for plumbing smokes,
not as a drop-in control for current CWM prompt experiments.

### ControlNet experiment caches

Configs refer to:

```text
/data/binghe/h3_proxy/cache/gta_v2_ctrl
/data/binghe/h3_proxy/cache/gta_v2_ctrl_wn
```

Treat these as experiment-specific and verify their existence and metadata
before use. They train a control trunk and are not interchangeable with the
reference-prefix LoRA caches above.

## ABot proxy corpus

### Legacy precomposed-DUV generation

Raw root:

```text
/data/binghe/datasets/ABot-sub-2000-clips
```

Known manifests and validation:

```text
/data/binghe/h3_proxy/abot_train.jsonl
/data/binghe/h3_proxy/abot_train_v2.jsonl
/data/binghe/h3_proxy/abot_validation_val6.json
```

Known preencoded cache:

```text
/data/binghe/h3_proxy/cache/abot_train
```

Historical manifest counts were approximately:

- `abot_train.jsonl`: 9,865 rows;
- `abot_train_v2.jsonl`: 9,545 rows.

Recount before use; those figures came from an earlier preparation run rather
than a frozen release receipt.

ABot clips use 124 frames at 24 FPS, a 1344 x 768 RGB target, and a 336 x 192
precomposed DUV video. Its DUV delivery convention differs from the GTA/CWM
packed DUV convention. Do not mix ABot and GTA proxy tensors merely because
their shapes agree.

### MoGe3 + SAM2 CWM12 generation

This is the new, quality-filtered ABot generation:

```text
/data/binghe/datasets/ABot-sub-2000-clips-moge3
```

Each clip retains the common 124-frame, 24-FPS, approximately 5.17-second
timeline. The photoreal target is 768 x 1344. The per-frame proxy deliverable
is 192 x 336 raw CWM12 DUV: metric MoGe3 depth plus SAM2-refined semantic
classes. Semantic sky pixels have zero depth. Do not substitute the legacy
`proxy/duv.mp4`; its Standard11/depth packing does not match this generation.

The complete DUV audit covered all 9,985 clips:

```text
audited:       9985
failed:           0
median_spread: 2.227
warnings:         0
notices:        463
```

The 463 notices are clips with no sky pixels and are non-blocking. Earlier
finite-depth sky surfaces were repaired before this audit.

Caption preparation found 9,721 captioned clips, including 61 failed and 893
warned captions; 264 clips had no caption. The mean caption score was 0.944.
The strict training policy excludes missing, failed, and warned captions,
requires score >= 0.90, and requires a successful DUV audit.

The split is by whole source episode, so one episode cannot leak between
training and validation:

```text
source clips:       9985
eligible clips:     8335
train clips:        8238
validation clips:     97
validation episodes:  24
```

Split artifacts:

```text
/data/binghe/datasets/ABot-sub-2000-clips-moge3/_fastvideo/train.jsonl
/data/binghe/datasets/ABot-sub-2000-clips-moge3/_fastvideo/val.jsonl
/data/binghe/datasets/ABot-sub-2000-clips-moge3/_fastvideo/split_summary.json
```

The active training-cache destination is:

```text
/data/binghe/h3_proxy/cache/abot_moge3_sam2_w0_qwen2
```

Its contract is:

- FastVideo `.pt` proxy cache, one sample per file;
- all 124 target and proxy frames are VAE-encoded;
- target latent geometry 37 x 48 x 84;
- proxy latent geometry 37 x 12 x 21;
- target frame zero is the picture anchor, resized to short edge 2048;
- reference order `<Picture 1>` then `<Video 1>`;
- Qwen samples `<Video 1>` at 2 FPS only when creating prompt embeddings;
- CWM system `w0`, one given target-latent frame;
- proxy time alignment disabled;
- no camera-conditioning contract.

Qwen 2 FPS does not reduce the VAE timeline. Never merge this directory with
the older Qwen-24-FPS ABot cache. Before training, verify that all 8,238
expected `.pt` samples are present and that their embedded metadata reports
Qwen 2 FPS and CWM `w0`.

The 97 held-out clips are not an in-training validation stream. SolarWM proxy
training intentionally disables native camera-conditioned validation; encode
the held-out manifest with the same cache contract and run the separate proxy
inference path when qualitative evaluation is required.

## Native SolarWM data

SolarWM's released MiniMax-H3 latent generation is:

```text
minimax-h3-158f-768p-nomind-v1
```

The public inventory contains 686,841 samples. The overseas deployment
runbook places it at:

```text
/mnt/solar/data/SolarWM-Data/releases-v1/
  latent-wds/minimax-h3-158f-768p-nomind-v1/
```

Matching recipe indexes live under:

```text
recipes/clean-158f-h3/latent-wds/
  minimax-h3-158f-768p-nomind-v1/
    train-index.jsonl.gz
    test-index.jsonl.gz
```

Required support artifacts:

```text
latent-wds/minimax-h3-158f-768p-nomind-v1/support/
  encoder_contract.json
  h3_silence_153_158_170.safetensors
```

The SolarWM H3 preencode contract is `h3.158f.v1`:

- 158 contiguous RGB frames at 768 x 1344;
- 47 target latent frames, tensor shape `[24, 47, 48, 84]`;
- one RGB anchor latent, `[24, 1, 48, 84]`;
- Qwen prompt embedding `[L, 5120]` and token tags `[L]`;
- 158 absolute `c2w` matrices and normalized intrinsics;
- WebDataset shards plus ordered, release-relative indexes;
- no DUV proxy latent in the current native schema.

Use this generation for unmodified SolarWM Stage0.5, Stage1, and Stage2
experiments.

## FastVideo proxy cache in SolarWM

SolarWM now has a separate `h3.ref2va-proxy.124f.v1` contract that directly
reads these FastVideo `.pt` caches for Stage0.5 training and proxy inference:

- 124 pixel frames and 37 target latent frames;
- one `.pt` sample containing `vae_latent`, `proxy_latent`, `anchor_latent`,
  prompt embeddings, token tags, and cache metadata;
- `<Picture 1>` followed by `<Video 1>`;
- W0 or WN given-frame behavior derived from the cached CWM role;
- no camera conditioning.

This support does not turn the cache into native `h3.158f.v1` data. Native
SolarWM remains a 158-frame, 47-latent, indexed-WebDataset contract with
authoritative camera matrices and no proxy tensor. Do not use a proxy cache
with native Stage0.5, Stage1, or Stage2 configs, and do not repack it into tar
files while claiming native compatibility.

## Recommended SolarWM experiment routes

### Route A: use SolarWM unchanged

Use the published 158-frame H3 latent-WDS. This is the immediate route for
testing SolarWM optimization, LoRA behavior, camera control, checkpointing,
Stage1, or Stage2 without proxy conditioning.

### Route B: move GTA/ABot RGB data into native SolarWM

Rebuild raw samples with:

- 158 contiguous source frames;
- 768 x 1344 RGB;
- simple or detailed caption;
- exact source-frame indices;
- absolute camera `c2w`;
- normalized camera intrinsics.

Then create a SolarWM raw-WDS/index and run the native H3 preencoder. Existing
124-frame clips cannot satisfy this contract without returning to the longer
source sequences or deliberately adding a new 124-frame SolarWM geometry.

This route can compare simple versus detailed captions, but it drops the DUV
proxy because the native backend does not consume it.

### Route C: use the isolated Ref2VA proxy profile

Use `data.input_mode=proxy_preencoded`,
`data.preencode_version=h3.ref2va-proxy.124f.v1`, and
`data.format=fastvideo_pt`. The reader validates latent geometry, reference
order, Qwen FPS, CWM role, given-frame count, and time-alignment policy before
training. The checked-in Stage0.5 proxy example is the starting point; override
its GTA cache and output paths for ABot.

Keep in-training validation disabled. Proxy visualization uses SolarWM's
separate inference action and must preserve the exact cache contract.

## Verification commands

### Count manifests and cache membership

```bash
wc -l \
  /data/binghe/h3_proxy/gta_v2_cwm_encode.jsonl \
  /data/binghe/h3_proxy/gta_v2_simple_qwen2_train.jsonl \
  /data/binghe/h3_proxy/gta_v2_detailed_qwen2_train.jsonl

for cache in \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24 \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24_wn \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_detailed
do
  printf '%s  ' "$cache"
  python - "$cache" <<'PY'
from pathlib import Path
import sys
print(len(list(Path(sys.argv[1]).glob("*.pt"))))
PY
done
```

### Read cache contracts

```bash
cd /workspace/FastVideo

python scripts/h3_proxy/describe_cache.py \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24 \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen24_wn \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple \
  /data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_detailed
```

### Assert the prompt arms have identical sample membership

```bash
python - <<'PY'
from pathlib import Path

simple = Path("/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple")
detailed = Path("/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_detailed")
a = {path.name for path in simple.glob("*.pt")}
b = {path.name for path in detailed.glob("*.pt")}
assert a == b, f"simple-only={len(a-b)}, detailed-only={len(b-a)}"
print(f"matched cache membership: {len(a)} clips")
PY
```

### Assert validation differs only by caption

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("/data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json"),
    Path("/data/binghe/h3_proxy/gta_v2_colleague_validation_val6.json"),
]
left, right = [json.loads(path.read_text())["data"] for path in paths]
assert len(left) == len(right)
for a, b in zip(left, right, strict=True):
    assert {k: v for k, v in a.items() if k != "caption"} == {
        k: v for k, v in b.items() if k != "caption"
    }
print(f"matched validation media: {len(left)} clips; caption is the only difference")
PY
```

### Verify a SolarWM native H3 installation

```bash
export SOLAR_DATA_ROOT=/mnt/solar/data/SolarWM-Data/releases-v1
export H3_GENERATION="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1"
export H3_SUPPORT="$H3_GENERATION/support"

test -r "$H3_SUPPORT/encoder_contract.json"
test -r "$H3_SUPPORT/h3_silence_153_158_170.safetensors"
test -r "$SOLAR_DATA_ROOT/recipes/clean-158f-h3/latent-wds/minimax-h3-158f-768p-nomind-v1/train-index.jsonl.gz"
test -r "$SOLAR_DATA_ROOT/recipes/clean-158f-h3/latent-wds/minimax-h3-158f-768p-nomind-v1/test-index.jsonl.gz"
```

## Experiment hygiene

- Record raw root, manifest digest, cache path, validation JSON, system role,
  Qwen FPS, reference order, given-frame count, and alignment policy with every
  run.
- Never infer a cache contract from its directory name alone; inspect sample
  metadata.
- Do not compare two prompt arms unless media membership and all non-caption
  validation fields match.
- Do not call an evaluation “checkpoint 300” when the completed training
  horizon was 285; use the actual complete checkpoint directory.
- A checkpoint directory without `dcp/.metadata` is incomplete and cannot be
  resumed.
- Training-time validation media upload and standalone evaluation upload are
  separate settings. Keep training validation disabled for the current
  FastVideo runs; standalone evaluations may upload to W&B.
