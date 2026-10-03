# MiniMax-H3 Ref2VA proxy：训练“训不动”排查清单

本文用于排查 SolarWM 和 FastVideo 上的 MiniMax-H3 Ref2VA proxy LoRA 实验。目标不是仅凭
loss 曲线判断，而是依次证明：

1. 数据和 prompt 合同与实验设定一致；
2. LoRA 是唯一可学习参数，并且被 optimizer 持有；
3. 梯度经过 SP/DP 同步后非零且有限；
4. FP32 master weight 在执行更新；
5. checkpoint 中的 live LoRA 确实离开初始化；
6. evaluation 确实加载了目标 checkpoint，而不是 base 或严重滞后的 EMA；
7. 固定输入和随机性后，输出随 checkpoint 发生可复现变化。

只有第 1–7 项全部成立后，才能把“效果不好”归因于数据、目标函数、adapter 容量或模型能力。
单独看到 loss 在 `0.1` 左右波动，不能证明模型没有学习。

相关资料：

- 数据和 prompt 合同：`docs/h3-proxy-dataset-inventory.md`
- H3 backend 与 proxy 用法：`docs/backends/minimax-h3.md`
- 两节点环境：`docs/runbooks/h3-2x8-h200-overseas.md`

## 1. 当前实验基线

### 1.1 共同的数据合同

当前 GTA/ABot proxy 实验应保持：

- target：124 个 24-FPS 像素帧，VAE latent 为 `[24, 37, 48, 84]`；
- proxy：完整 124 帧参与 VAE，latent 为 `[24, 37, 12, 21]`；
- Qwen 只以 2 FPS 观看 `<Video 1>`，这不会降低 VAE 时间采样率；
- reference 顺序为 `<Picture 1>` 后接 `<Video 1>`；
- CWM system role 为 `w0`；
- `num_given_latent_frames=1`，首个 target latent 被 anchor 硬锁；
- `align_proxy_reference_time=false`；
- anchor 来自 target 第 0 帧，short edge 为 2048；
- 不启用 camera conditioning；
- Simple/Detailed 指 user caption，不表示移除 CWM system prompt。

当前主要 cache：

```text
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple
/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_detailed
/data/binghe/h3_proxy/cache/abot_moge3_sam2_w0_qwen2
```

不要仅根据目录名判断合同，必须读取 `.pt` 内的 `info` 和 tensor shape。

### 1.2 当前 FastVideo 对照组

当前 FastVideo fresh 3000-step LoRA 对照使用：

- 单节点 8 GPU，SP=1；
- micro batch=1，gradient accumulation=1，global batch=8；
- rank/alpha=128/128；
- 50 个主 transformer block 的 Q/K/V/out 和 FFN in/out；
- 300 个 target linear modules、600 个 LoRA A/B tensor；
- peak LR `2e-5`；
- warmup 100 steps，之后 cosine，minimum ratio `0.05`；
- 每 300 steps 保存；
- 训练期间不执行 validation，不上传 media；
- W&B loss 是全 DP 平均，并额外记录 100-step rolling mean。

### 1.3 当前 SolarWM proxy 组

配置：

```text
configs/examples/minimax_h3/stage0p5-124f-ref2va-proxy-sp1.yaml
```

当前 3000-step 启动覆盖项为：

```text
train.max_steps=3000
checkpoint.save_every_steps=250
validation.manual_steps=[500,1000,1500,2000,2500,3000]
```

其训练设定为：

- 两节点 × 8 GPU，raw world size=16；
- SP=1，logical DP=16；
- micro batch=1，gradient accumulation=1，global batch=16；
- rank/alpha=128/128；
- 只训练 50 个主 transformer block 的 Q/K/V/out；
- 200 个 target linear modules、400 个 LoRA A/B tensor；
- live LoRA 参数为 BF16；
- optimizer 使用 FP32 master weight 和 FP32 AdamW moments；
- peak LR `2e-5`；
- warmup 10 steps，之后 cosine，minimum ratio `0.05`；
- EMA decay `0.9999`，从 step 0 每步更新；
- validation/media 不在训练进程中执行；
- loss 在 logical DP group 上平均，W&B 同时记录 loss EMA，默认 beta `0.95`。

这不是 FastVideo 组的严格框架对照。至少存在四个变量：

- global batch：16 对 8；
- warmup：10 对 100；
- LoRA target：attention-only 对 attention+FFN；
- SolarWM evaluation 可选择 live 或 EMA，而 FastVideo 通常直接评估 live LoRA。

因此，在这些变量没有对齐前，不能根据视觉结果直接断言“SolarWM/FastVideo 训不动”。

## 2. 当前最高优先级风险：evaluation 使用了 EMA

SolarWM proxy inference 配置默认：

```yaml
checkpoint:
  weight_source: ema
```

当前批量 evaluation 命令也显式传了：

```text
--set checkpoint.weight_source=ema
```

这对检查最终平滑模型有意义，但不适合首先证明 LoRA 是否训得动。EMA decay 为 `0.9999`，且没有
对“相对 fresh LoRA 初始化的位移”做 bias correction。对于一次持续不变的权重位移，EMA 在第
`n` 次更新后只吸收约 `1 - 0.9999^n`：

- step 500：约 4.88%；
- step 1000：约 9.52%；
- step 1500：约 13.93%；
- step 3000：约 25.92%。

真实训练权重一直在变化，不能把这些百分比当作精确 live/EMA 比例，但它说明 EMA 会显著滞后。

排查训练是否有效时：

1. 先用 `checkpoint.weight_source=live`；
2. 再把相同 checkpoint 用 `ema` 作为独立对照；
3. 不要只看 EMA 后得出“LoRA 没变化”的结论。

## 3. 启动前清单

### 3.1 固化代码和配置身份

- [ ] 两节点使用同一个 commit；
- [ ] 镜像、PyTorch、CUDA、PEFT 版本一致；
- [ ] `runtime.output_dir` 是全新目录，或明确指向要 resume 的旧目录；
- [ ] 保存最终 resolved config，不只保存手写命令；
- [ ] 记录数据目录/manifest 的 digest；
- [ ] 不复用旧 W&B run ID 做一个语义不同的实验。

先解析并验证最终配置：

```bash
cd /workspace/SolarWM

CONFIG=configs/examples/minimax_h3/stage0p5-124f-ref2va-proxy-sp1.yaml

python -m solarwm config resolve \
  --config "$CONFIG" \
  --set train.max_steps=3000 \
  --set checkpoint.save_every_steps=250 \
  --set validation.manual_steps='[500,1000,1500,2000,2500,3000]' \
  --set runtime.output_dir=/data/binghe/h3_proxy/solarwm-runs/gta-v2-w0-lora128-3000 \
  --output /tmp/h3-proxy-resolved.json
```

检查关键字段：

```bash
python - /tmp/h3-proxy-resolved.json <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))["config"]
assert cfg["model"]["transformer_subfolder"] == "transformer_ref"
assert cfg["model"]["training_mode"] == "lora"
assert cfg["model"]["conditioning_mode"] == "ref2va_proxy"
assert cfg["model"]["adapter"]["target"] == "main_attention_qkvo"
assert cfg["model"]["adapter"]["rank"] == 128
assert cfg["model"]["adapter"]["alpha"] == 128
assert cfg["data"]["input_mode"] == "proxy_preencoded"
assert cfg["data"]["qwen_video_fps"] == 2
assert cfg["data"]["cwm_system"] == "w0"
assert cfg["data"]["num_given_latent_frames"] == 1
assert cfg["data"]["align_proxy_reference_time"] is False
assert cfg["train"]["global_batch_size"] == 16
assert cfg["train"]["optimizer"]["name"] == "fp32_master_adamw"
assert cfg["train"]["optimizer"]["learning_rate"] == 2e-5
assert cfg["validation"]["validate_every_steps"] == 0
assert cfg["validation"]["smoke_step"] == 0
assert cfg["runtime"]["tracking"]["log_media"] is False
print("resolved H3 proxy contract: OK")
PY
```

### 3.2 检查 cache 合同

```bash
export CACHE=/data/binghe/h3_proxy/cache/gta_v2_cwm_1344_qwen2_simple

python - "$CACHE" <<'PY'
from collections import Counter
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
paths = sorted(root.glob("*.pt"))
assert paths, f"no .pt files under {root}"
chosen = paths[::max(1, len(paths) // 64)][:64]
contracts = Counter()

for path in chosen:
    sample = torch.load(path, map_location="cpu", weights_only=True)
    info = sample["info"]
    contract = (
        tuple(sample["vae_latent"].shape),
        tuple(sample["proxy_latent"].shape),
        int(sample["anchor_latent"].shape[-3]),
        min(sample["anchor_latent"].shape[-2:]) * 16,
        int(info["num_frames"]),
        float(info["qwen_video_fps"]),
        str(info["cwm_system"]),
    )
    contracts[contract] += 1
    assert str(info.get("prompt", "")).strip(), path

assert len(contracts) == 1, contracts
contract = next(iter(contracts))
assert contract[:2] == ((24, 37, 48, 84), (24, 37, 12, 21))
assert contract[2:] == (1, 2048, 124, 2.0, "w0")
print({"clips": len(paths), "sampled": len(chosen), "contract": contract})
PY
```

ABot active cache 预期为 8,238 个 `.pt`。GTA 以实际 cache 文件数为准，不能使用未过滤 manifest
的 771 行代替成功编码的样本数。

SolarWM 当前 `data.data_path` 接受一个 cache 目录或一个文本 manifest。它不接受 FastVideo 的
YAML list/dict 混合写法。要做 GTA+ABot 自然混合，应生成一份每行一个绝对 `.pt` 路径的不可变
manifest，记录行数和 digest，再把 `data.data_path` 指向它。

### 3.3 检查拓扑和 batch

必须满足：

```text
raw world size = NNODES × NPROC_PER_NODE
logical DP size = raw world size / SP size
global batch = logical DP size × micro batch × gradient accumulation
```

当前 SolarWM proxy 组为：

```text
16 / 1 × 1 × 1 = 16
```

- [ ] 两节点均看到 `WORLD_SIZE=16`；
- [ ] node rank 分别为 0/1；
- [ ] rendezvous ID、地址和端口一致；
- [ ] 两节点能读相同 cache、模型和 output path；
- [ ] 训练日志确认没有把 16-GPU 配置启动成单节点 8 进程。

### 3.4 检查 LoRA/FSDP/optimizer 链路

SolarWM 当前实现的正确顺序是：

```text
load transformer_ref
  -> freeze whole base
  -> inject PEFT LoRA
  -> audit all-and-only LoRA requires_grad=True
  -> FSDP wrap frozen base
  -> keep LoRA in ignored_states (replicated)
  -> build FP32MasterAdamW from exactly those LoRA parameters
```

精度合同：

- base 和 live LoRA 可以是 BF16；
- FSDP `param_dtype=null` 不会强制把参数变成 FP32；
- FSDP reduction 为 FP32；
- replicated LoRA 在 optimizer step 前显式执行 SP sum、DP average，bucket 临时转 FP32；
- gradient clipping 在同步后执行；
- AdamW master weight、`exp_avg`、`exp_avg_sq` 均为 FP32；
- optimizer 更新 FP32 master 后再复制回 BF16 live LoRA。

因此，“FSDP 只支持 FP32 trainable parameters”不是这里的要求。真正的验收项是 FP32 reduction
和 FP32 optimizer master 是否存在。

## 4. 先跑 3-step 机械 smoke

在 3000-step 正式运行前，先用相同数据、相同拓扑跑 3 steps：

```text
train.max_steps=3
checkpoint.save_every_steps=3
validation.manual_steps=[3]
runtime.tracking.enabled=false
runtime.output_dir=<全新的 smoke 目录>
```

验收：

- [ ] 三个 optimizer step 全部完成；
- [ ] 每步 loss 有限；
- [ ] gradient norm 有限且不是连续的精确零；
- [ ] LR 非零并按 warmup 改变；
- [ ] `checkpoint_model_000003/COMPLETE.json` 存在；
- [ ] checkpoint manifest 验证通过；
- [ ] LoRA-B 已离开零初始化；
- [ ] optimizer state 含 FP32 master/moments，step 为 3。

smoke 失败时不要直接启动 3000 steps。

## 5. 训练中检查 loss、梯度和 LR

SolarWM 的 `training-events.jsonl` 是 rank-0 的权威事件流。proxy runtime 会先把 loss 在 logical
DP group 上平均，再写事件；它不是 rank-0 单样本 loss。W&B：

- `train/loss/*`：当前全 DP 平均 loss；
- `train/loss_ema/*`：日志平滑曲线，不是模型 EMA；
- `train/gradient_norm`：同步后、裁剪前 norm；
- `train/learning_rate`：scheduler step 后的 LR。

快速检查：

```bash
export RUN=/data/binghe/h3_proxy/solarwm-runs/gta-v2-w0-lora128-3000

python - "$RUN/training-events.jsonl" <<'PY'
import json
import math
import statistics
import sys

events = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
steps = [event for event in events if event.get("event") == "optimizer_step"]
assert steps, "no optimizer_step events"

loss_key = next(iter(steps[0]["losses"]))
losses = [float(event["losses"][loss_key]) for event in steps]
grads = [float(event["gradient_norm"]) for event in steps]
lrs = [float(event["lr"]) for event in steps]
assert all(math.isfinite(value) for value in losses + grads + lrs)

window = min(100, len(losses))
print({
    "steps": (steps[0]["step"], steps[-1]["step"], len(steps)),
    "loss_key": loss_key,
    "loss_first_window": statistics.fmean(losses[:window]),
    "loss_last_window": statistics.fmean(losses[-window:]),
    "grad_min": min(grads),
    "grad_median": statistics.median(grads),
    "grad_max": max(grads),
    "exact_zero_grad_steps": sum(value == 0.0 for value in grads),
    "lr_first_last": (lrs[0], lrs[-1]),
})
PY
```

解释边界：

- flow matching 每步重采样样本、noise 和 timestep，raw loss 本来就会强烈波动；
- loss 长期不单调下降不等于梯度断开；
- loss EMA/100-step mean 适合看趋势，但也不能替代 checkpoint 权重和固定种子 evaluation；
- gradient norm 非零只证明反向链路存在，不证明 optimizer 更新了正确参数；
- gradient norm 大量被 clip 到阈值附近时，应单独检查 LR、目标函数和数据异常。

## 6. 验证 checkpoint 真的包含更新

先确认 checkpoint 完整：

```bash
export CKPT=/data/binghe/h3_proxy/solarwm-runs/gta-v2-w0-lora128-3000/checkpoint_model_000500

test -f "$CKPT/COMPLETE.json"
test -f "$CKPT/checkpoint-manifest.json"
test -f "$CKPT/adapter.pt"
test -f "$CKPT/optimizer.pt"
test -f "$CKPT/ema.pt"
test -f "$CKPT/runtime.pt"
```

检查 live LoRA 和 optimizer：

```bash
python - "$CKPT" <<'PY'
from pathlib import Path
import math
import sys
import torch

root = Path(sys.argv[1])
adapter = torch.load(root / "adapter.pt", map_location="cpu", mmap=True, weights_only=True)
runtime = torch.load(root / "runtime.pt", map_location="cpu", weights_only=True)
optimizer = torch.load(root / "optimizer.pt", map_location="cpu", mmap=True, weights_only=True)

metadata = adapter["metadata"]
state = adapter["state"]
a = [value for key, value in state.items() if ".lora_A." in key]
b = [value for key, value in state.items() if ".lora_B." in key]
assert metadata["target_count"] == 200, metadata["target_count"]
assert len(a) == len(b) == 200, (len(a), len(b))
assert all(value.dtype == torch.bfloat16 for value in a + b)

b_norm = math.sqrt(sum(float(value.float().square().sum()) for value in b))
b_max = max(float(value.float().abs().max()) for value in b)
assert b_norm > 0.0 and b_max > 0.0, "LoRA-B is still zero"

slots = list(optimizer["state"].values())
assert slots, "optimizer has no parameter state"
for slot in slots:
    assert slot["master_param"].dtype == torch.float32
    assert slot["exp_avg"].dtype == torch.float32
    assert slot["exp_avg_sq"].dtype == torch.float32
steps = {int(slot["step"].item()) for slot in slots}

print({
    "checkpoint_step": int(runtime["global_step"]),
    "target_modules": metadata["target_count"],
    "trainable_tensors": len(state),
    "trainable_parameters": metadata["trainable_parameters"],
    "live_dtype": sorted(set(metadata["state_dtypes"].values())),
    "lora_B_norm": b_norm,
    "lora_B_max_abs": b_max,
    "optimizer_parameter_states": len(slots),
    "optimizer_steps": sorted(steps),
})
PY
```

判断：

- `LoRA-B > 0`：权重已经离开 fresh LoRA 的零输出初始化；
- optimizer step 与 checkpoint step 一致：不是只保存了权重而 optimizer 没走；
- FP32 master/moments 存在：BF16 live 参数并未迫使 AdamW 在 BF16 内累计；
- 这些结论证明“机械上训得动”，仍不证明视觉方向正确。

## 7. live 与 EMA 必须分开评估

对同一个 checkpoint、同一个 cache、同一组 sample、同一 seed，至少运行：

```text
base: checkpoint.resume_from=null
live: checkpoint.resume_from=<checkpoint>; checkpoint.weight_source=live
ema:  checkpoint.resume_from=<checkpoint>; checkpoint.weight_source=ema
```

排查时优先运行 live：

```bash
--set checkpoint.resume_from="$CKPT" \
--set checkpoint.weight_source=live
```

EMA 作为第二组：

```bash
--set checkpoint.resume_from="$CKPT" \
--set checkpoint.weight_source=ema
```

必须固定：

- cache 和 sample IDs；
- caption/text embedding；
- picture anchor 和 video proxy；
- CWM role；
- given-frame count；
- proxy time-alignment；
- seed/noise；
- sampling steps；
- guidance；
- first-frame hard lock；
- output codec 和 panel 顺序。

不要拿不同 prompt、不同 proxy、不同 seed 的视频判断 checkpoint 是否移动。

## 8. 结果决策树

### 情况 A：没有 optimizer-step 事件

训练没有进入更新环。检查 rendezvous、数据 reader、模型加载、FSDP 初始化和 OOM。不要讨论 loss 或
adapter 容量。

### 情况 B：gradient norm 为零或没有 LoRA gradient

反向图或可学习参数有问题。检查：

- LoRA 是否在 forward 使用的 `transformer_ref` 上；
- `requires_grad=True` 是否只落在 400 个 LoRA tensor；
- conditioning 是否被 detach 或绕过；
- 所有 rank 的 gradient pattern 是否一致；
- `sync_lora_gradients` 是否在 clipping/optimizer 前执行。

SolarWM 当前代码会在完全没有 LoRA gradient 或 rank 间 gradient pattern 不一致时直接报错，而不是
静默继续。

### 情况 C：gradient norm 非零，但 LoRA-B 仍为零

optimizer 没有更新 live LoRA，或 checkpoint 保存了错误对象。检查 optimizer 参数 identity、
`optimizer.step()`、FP32 master 回写和 checkpoint `adapter.pt`。

### 情况 D：live LoRA-B 非零，live evaluation 与 base 字节级相同

优先怀疑 evaluation 没有加载目标权重，或输出目录复用了旧结果。确认：

- `checkpoint.resume_from` 指向完整 checkpoint；
- `checkpoint.weight_source=live`；
- inference receipt 中的 checkpoint digest 和 step 正确；
- `proxy-inference/COMPLETE.json` 属于本次输出；
- 输出目录在启动前不存在；
- 没有把 base baseline 和 checkpoint 结果写进同一路径。

### 情况 E：live evaluation 有变化，EMA 几乎没变化

训练链路正常，主要是 `0.9999` EMA 滞后。继续把 live 作为 trainability 诊断；EMA decay 是否调整是
另一个实验变量。

### 情况 F：live 和 EMA 都有变化，但控制或画质变差

这不是“训不动”。转向检查：

- attention-only LoRA 容量是否不足；
- 是否要像 FastVideo 当前组一样加入 FFN；
- LR、global batch、warmup 是否匹配；
- proxy 对目标 motion 的可辨识度；
- flow-matching loss 是否过度奖励平均化/模糊结果；
- 数据中的 target/proxy/prompt 是否互相矛盾；
- 3000 steps 是否过拟合小型 GTA cache。

### 情况 G：loss 无明显下降，但固定 evaluation 持续变化

以固定 evaluation 和 checkpoint drift 为准。随机 timestep/noise 下的 loss 平台不等于没有学习。

## 9. 当前实验的最小评价矩阵

每次至少保留：

- base；
- step 3 live（机械 smoke）；
- step 500 live；
- step 1000 live；
- step 1500 live；
- step 2000 live；
- step 2500 live；
- step 3000 live；
- step 500/1500/3000 EMA 作为平滑对照。

如果资源有限，不能省略 base、step 3 live、最终 live。只看最终 EMA 无法区分：

- adapter 没更新；
- adapter 更新但 evaluator 没加载；
- live 已变化但 EMA 严重滞后；
- 模型确实更新但学习方向不好。

## 10. 每个 run 必须记录

```text
repository commit:
container/image digest:
base model path and identity:
config source digest:
resolved config digest:
cache path or manifest:
cache/manifest digest:
sample count:
prompt arm:
CWM role:
Qwen video FPS:
reference order:
given latent frames:
proxy time alignment:
LoRA target/rank/alpha:
live LoRA dtype:
optimizer/master dtype:
world/SP/DP:
micro batch:
gradient accumulation:
global batch:
peak LR:
warmup:
schedule/min ratio:
max steps:
save steps:
EMA decay:
evaluation weight source:
evaluation sample IDs:
evaluation seed/sampling/guidance:
W&B run ID:
output directory:
```

缺少其中任一关键 conditioning 或 evaluation 字段时，不应把两个结果称为严格对照。
