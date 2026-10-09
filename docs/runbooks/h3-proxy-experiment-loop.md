# MiniMax-H3 proxy：单节点并行实验与滚动评估

`python -m solarwm.backends.minimax_h3.proxy_loop` 把一个 8×H200 节点当作一个独立实验槽：
每个节点用 `local_file_launcher`（文件 rendezvous，不依赖 `torchrun --standalone`）训练一个
SP1、global batch 8 的实验，另一个节点上的 `eval-worker` 按 checkpoint 落盘顺序做推理。

## 前提

- 代码：`/workspace/SolarWM`（`public-h3-image`），在 `/opt/venv` 中 editable 安装。
- W&B key：`/data/binghe/.secrets/wandb_api_key`。环境里已有 `WANDB_API_KEY` 时优先使用环境变量。
- 每个命令都会 source `/usr/local/gib/scripts/set_nccl_env.sh`，并设置
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；启动前会顺序读一遍
  `transformer_ref/*.safetensors`，让 8 个 rank 从 page cache 加载权重（`--no-warm` 跳过）。

## 训练

```bash
cd /workspace/SolarWM
python -m solarwm.backends.minimax_h3.proxy_loop train \
  --name omni-med-unaligned-warm \
  --config configs/examples/minimax_h3/stage0p5-124f-ref2va-omni-mixed-low-704p-sp1.yaml \
  --set data.data_path=/data/binghe/h3_proxy/cache/omni-mixed-med-704p-qwen2 \
  --set data.proxy_latent_height=22 --set data.proxy_latent_width=40 \
  --set checkpoint.initialization.student.path=/data/.../checkpoint_model_003000
```

- 会自动覆盖 `distributed.world_size=8`、`train.global_batch_size`（按
  `nproc / sp × micro × accum` 计算）、`runtime.output_dir=<run-root>/<name>`、
  `runtime.tracking.run_name=<name>`；用户的 `--set` 排在最后，同名时以用户为准。
- `--resume auto`（默认）从最新带 `COMPLETE.json` 的 checkpoint 完整续训；`--resume never` 关闭。
  已有 `run-result.json` 的 run 默认拒绝再启动。
- `--dry-run` 只加载并校验最终配置并打印覆盖项，不占 GPU。

## 滚动评估

```bash
python -m solarwm.backends.minimax_h3.proxy_loop eval-worker \
  --loop round1 --runs omni-a omni-b omni-c \
  --infer-config configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml \
  --eval-data /data/binghe/h3_proxy/cache/<held-out-cache> \
  --ablations correct static
```

- checkpoint 以 `training-events.jsonl` 的 checkpoint 事件和目录中的 `COMPLETE.json` 为准。
- 每个任务的 infer 配置由 run 自己的 `resolved-config.json` 的 `model` 与 `data` 段生成，
  因此对齐方式、proxy 分辨率和模态集合天然与训练一致；`--eval-data` 只替换 `data_path`。
- 队列按 step 从小到大、run 轮转排序，让每个实验尽早获得反馈。
- 推理写到 `/workspace/h3loop/evals/<loop>/...`，结束后复制到
  `/data/binghe/h3_proxy/evals/loops/<loop>/<run>/step-XXXXXX/<ablation>/`，成功写 `DONE.json`；
  失败计入 `failures.json`，默认最多重试 2 次。
- W&B：项目沿用 infer 配置（`solarwm-h3-proxy-eval`），group 为 `--loop`，run 名为
  `<run>-s<step>-<ablation>`。
- 所有 run 都有 `run-result.json` 且队列为空时退出；`--once` 处理完当前队列即退出；
  `--plan` 只列出并校验待评估任务。

## 查看结果

```bash
python -m solarwm.backends.minimax_h3.proxy_loop status --runs omni-a omni-b omni-c --loop round1
python -m solarwm.backends.minimax_h3.proxy_loop frames --loop round1 --run omni-a --step 500
```

`status` 输出 JSON：步数、bias-corrected loss EMA、窗口均值、lr、grad norm、近 20 步耗时与峰值
显存、已完成 checkpoint 和已完成评估。`frames` 把每个 `compare.mp4` 等间隔取 5 帧纵向拼成 PNG，
写到 `/workspace/h3loop/frames/...`，便于拉回本地查看。
