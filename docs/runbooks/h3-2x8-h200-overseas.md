# SolarWM MiniMax-H3：2 × 8 H200 海外节点部署 Runbook

本文面向两台节点、每台 8 张 NVIDIA H200（共 16 GPU）的 SolarWM
MiniMax-H3 部署。覆盖宿主机验收、容器镜像、共享存储、模型与数据下载、
两机通信、训练/推理启动、验收、恢复和常见故障。

本文以当前 SolarWM 仓库中的 H3 版本为准：

- Python 3.10；
- PyTorch 2.6.0 + CUDA 12.4；
- torchvision 0.21.0；
- FlashAttention 2.8.3；
- Diffusers 0.40.0；
- Transformers 5.12.1；
- PEFT 0.20.0；
- BF16、FSDP 和 H3 LoRA-384。

## 0. 先确认三个部署决策

### 0.1 H3 许可证地域闸门

MiniMax H3 Community License 当前把欧盟、英国、韩国和美国列为
Excluded Territories。标准社区许可证不授权在这些地区使用、修改、分发或
展示 H3 Works 及其输出。

因此，在创建或下载任何 H3 权重前必须记录节点的物理地域：

```text
节点所在国家/地区：
云厂商及 region：
许可证审查人：
审查日期：
审查结论/单独授权编号：
```

如果节点位于上述排除地区，停止本 runbook，先通过
https://platform.minimax.io/h3-license 取得适用的单独授权。不要把“海外”
等同于“许可证允许”。许可证原文：
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

### 0.2 确定本次用途

按用途只下载必要资产：

- 仅做 Stage2 158 帧推理：基础模型、`-fix` Stage2 checkpoint、H3 latent
  generation 的 support 和测试索引。
- 做 Stage0.5 训练：基础模型、主数据仓库和完整 H3 latent generation。
- 做 Stage1 训练：再加 Stage0.5 EMA 初始化包；Stage1 验证还会读取 raw
  test camera，必须保证对应 raw-WDS 测试 shard 可读。
- 做 Stage2 训练：再加 Stage0.5 EMA、Stage1 EMA 两个初始化包。
- 做原始长度 Stage2 推理：再下载独立 test-set-v1（约 77.4 GB）。

不要下载完整 25 TB raw-WDS，除非确实要做 Stage1 完整验证、在线编码、
重新 preencode 或训练索引明确指向 raw-WDS。

### 0.3 本文的基础设施假设

以下命令假设：

- 两台 x86_64 Linux 节点，分别记为 `h3-node-0`、`h3-node-1`；
- 每台正好暴露 8 张完整 H200，不启用 MIG；
- 节点间有私网互通的 200/400 Gb/s InfiniBand 或 RoCE；
- 两节点能以完全相同路径访问模型、数据和输出目录；
- 使用 Docker/containerd；如使用 BCS/Kubernetes，见第 12 节；
- 两节点时间同步，主机名和私网地址稳定。

如果只有普通以太网，功能 smoke 可能通过，但 33B H3 的两机 FSDP 性能和
稳定性通常不可接受。正式训练前必须做跨节点 collective 压测。

## 1. 目标拓扑

物理拓扑：

```text
h3-node-0: GPU 0..7, NODE_RANK=0, rendezvous server
h3-node-1: GPU 0..7, NODE_RANK=1
NNODES=2
NPROC_PER_NODE=8
WORLD_SIZE=16
```

SolarWM 两机推荐参数：

- Stage0.5：SP=2，logical DP=16/2=8，global batch=8。
- Stage1：SP=2，logical DP=16/2=8，global batch=8。
- Stage2：SP=4，logical DP=16/4=4，global batch=4。
- 158 帧 Stage2 推理：一个 8-GPU 节点即可，SP=4。
- 原始长度 Stage2 推理：一个 8-GPU 节点，SP=8。

这里的两机训练 global batch 小于发布配方在 256 GPU 上的 batch。这会改变
优化语义，并不等价于官方发布训练结果；它是仓库 H3 文档明确给出的两机
缩放方式。

如果目标主要是推理，推荐把两台节点作为两个独立的 8-GPU worker，各跑一个
推理任务，而不是为单个样本跨节点通信。这样吞吐更高，故障域更小。

## 2. 宿主机和网络验收

### 2.1 驱动、Fabric Manager 和容器运行时

推荐宿主机：

- Ubuntu 22.04 LTS 或云厂商验证过的 H200 OS image；
- NVIDIA Data Center Driver R550 或更新的云厂商稳定版本；
- CUDA 12.4.1 容器要求驱动至少 550.54.15；
- HGX/NVSwitch 机器安装与驱动版本完全匹配的 `nvidia-fabricmanager`；
- NVIDIA Container Toolkit；
- Docker 24+ 或等价 containerd。

宿主机不需要安装 CUDA Toolkit；CUDA Toolkit 放在容器内。两台机器必须使用
同一驱动分支和同一 Fabric Manager 版本。

每台节点执行：

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi nvlink -s
systemctl is-active nvidia-fabricmanager || true
docker info
nvidia-container-cli info
```

验收条件：

- 每台恰好看到 8 张 H200，显存容量一致，无 MIG slice；
- `nvidia-smi` 无 Xid、ECC uncorrectable error；
- 节点内 GPU 经 NVLink/NVSwitch 连接，不应全部退化为 PCIe/系统互联；
- Fabric Manager 在需要它的 HGX 平台为 active；
- 两节点驱动版本一致。

用最小 CUDA 容器确认 GPU passthrough：

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi
```

### 2.2 RDMA 和 rendezvous

每台节点执行：

```bash
ip -br addr
ip route
ls -l /dev/infiniband || true
ibv_devinfo || true
ibdev2netdev || true
rdma link || true
```

记录：

```bash
export MASTER_ADDR=<h3-node-0 的训练私网 IP>
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=<训练私网接口，例如 bond0 或 eth1>
export NCCL_IB_HCA=<实际 mlx5 设备列表，例如 mlx5_0,mlx5_1>
```

不要照抄示例接口名。`NCCL_SOCKET_IFNAME` 指错是两机 hang 的高频原因。
RoCE 的 GID index、traffic class、PFC/ECN 必须使用云厂商给出的值；不要在
不知道网络配置时盲设 `NCCL_IB_GID_INDEX` 或 `NCCL_IB_TC`。

网络验收：

```bash
# node-0
nc -l 29500

# node-1
nc -vz "$MASTER_ADDR" 29500
```

生产安全组至少允许两节点在训练私网内双向通信，并允许 rendezvous port。
容器采用 host network 可避免 NCCL 动态连接被容器 NAT/端口映射破坏。不要把
29500 暴露到公网。

### 2.3 主机资源

每台节点建议：

- host RAM 至少 1 TB；以实际云规格和预检峰值为最终依据；
- 本地 NVMe 至少 2–4 TB，用于镜像层、HF/ModelScope cache、数据 cache 和
  临时文件；
- `/dev/shm` 至少 64 GB，推荐容器使用 host IPC；
- `memlock` unlimited；
- 关闭自动休眠和会中断长任务的维护策略；
- 时间同步误差小于 1 秒。

检查：

```bash
free -h
df -h
df -i
ulimit -l
timedatectl status
```

不要仅按 checkpoint 文件大小规划空间。训练输出包含 live/EMA、optimizer、
随机状态、验证视频和事务临时文件。输出文件系统应预留至少两个完整
checkpoint transaction 加验证产物的空间，并设置容量告警。

## 3. 存储布局

推荐统一路径：

```text
/mnt/solar/
|-- models/                 # 两节点只读
|-- data/
|   `-- SolarWM-Data/
|       `-- releases-v1/
|-- outputs/                # 两节点可写、强一致共享 POSIX 文件系统
|-- cache/
|   |-- hf/
|   |-- modelscope/
|   `-- torch/
`-- logs/
```

要求：

- `models/`、`data/`、`outputs/` 在两个容器中的绝对路径完全一致；
- `outputs/` 必须能被所有 rank 看到。不要让两台机器各自在本地创建同名
  输出目录，否则会得到不完整的分片 checkpoint；
- 数据可放共享高吞吐文件系统，也可完整复制到两节点相同路径。若复制，先
  比较文件数量、总字节数和抽样 checksum；
- 共享文件系统支持原子 rename、文件锁和 close-to-open consistency；
- 不要把高并发 WebDataset 训练直接压在低 IOPS 的小型 NFS 上。

创建目录：

```bash
sudo mkdir -p /mnt/solar/{models,data,outputs,cache/{hf,modelscope,torch},logs}
sudo chown -R "$(id -u):$(id -g)" /mnt/solar
```

如果模型和数据从共享盘只读挂载，下载只在一台管理节点执行一次。训练容器中
不要携带长期 Hugging Face 或 ModelScope token。

## 4. 容器镜像

需要从零构建、推送到 TCR 并取得可拉取 digest 时，直接按
[`h3-image-build-and-push.md`](h3-image-build-and-push.md) 操作。

### 4.1 推荐结论

正式环境不要直接使用浮动的“最新 PyTorch”镜像。

推荐基础镜像：

```text
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

在它上面构建项目内部镜像，锁定 Python 3.10 和所有 H3 包。FlashAttention
需要在有 CUDA compiler 的 `devel` 镜像中、安装 PyTorch 之后、关闭 build
isolation 编译。

官方现成镜像：

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

它已含 PyTorch 2.6.0/CUDA 12.4，但内置 Python 3.11，而 SolarWM H3 发布测试
基线是 Python 3.10。因此只建议用它做快速硬件/网络 smoke，不作为正式可复现
镜像。

### 4.2 构建正式 H3 镜像

在 SolarWM 仓库根目录临时创建以下 Dockerfile，或把同样内容放入内部镜像
构建系统：

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    HF_HOME=/mnt/solar/cache/hf \
    TORCH_HOME=/mnt/solar/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv \
      build-essential ninja-build git git-lfs ca-certificates curl \
      ffmpeg libgl1 libglib2.0-0 libibverbs1 ibverbs-providers rdma-core \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja

RUN python -m pip install \
      torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install \
      diffusers==0.40.0 \
      transformers==5.12.1 \
      peft==0.20.0 \
      imageio==2.37.4 \
      imageio-ffmpeg==0.6.0

WORKDIR /opt/SolarWM
COPY . /opt/SolarWM

RUN python -m pip install ".[train]"

ENV TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8
RUN python -m pip install --no-build-isolation flash-attn==2.8.3

RUN python -m solarwm environment probe \
    && python -m pip check

ENTRYPOINT []
CMD ["bash"]
```

构建时固定代码 commit：

```bash
cd /path/to/SolarWM
git status --short
git rev-parse HEAD

export REGISTRY=<内部镜像仓库>
export SOLARWM_COMMIT="$(git rev-parse --short=12 HEAD)"
export IMAGE="$REGISTRY/solarwm-h3:${SOLARWM_COMMIT}-torch2.6-cu124"

docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
docker build --pull -t "$IMAGE" -f /path/to/Dockerfile.h3 .
docker push "$IMAGE"
docker inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

将最终 image digest 写入变更单和 run manifest。两台节点都按 digest 拉取，避免
同一 tag 漂移：

```bash
docker pull "$IMAGE"
docker image inspect "$IMAGE" \
  --format '{{.Id}} {{json .RepoDigests}}'
```

如海外节点不能直连 Docker Hub，在有权限的构建区拉取、扫描并镜像到同 region
的私有 registry。不要使用未经验证的第三方镜像代理。

### 4.3 镜像内版本验收

```bash
docker run --rm --gpus all --ipc host \
  -v /mnt/solar:/mnt/solar \
  "$IMAGE" \
  bash -lc '
    python --version
    python -m solarwm environment probe
    python -m pip check
    python - <<PY
import torch
import flash_attn
import diffusers
import transformers
import peft
print("gpu_count", torch.cuda.device_count())
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash_attn", flash_attn.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
PY'
```

验收值应包括 Python 3.10、16 节点总 GPU 中本机 8 张、capability `(9, 0)`、
torch 2.6.0、CUDA 12.4 和仓库规定的包版本。

## 5. 启动长期运行容器

每台节点执行。先分别设置本机 rank（node-0 为 0，node-1 为 1）；RDMA device
名称按主机实际情况生成：

```bash
export NODE_RANK=<0-or-1>

RDMA_ARGS=()
if compgen -G "/dev/infiniband/*" >/dev/null; then
  for dev in /dev/infiniband/*; do
    RDMA_ARGS+=(--device "$dev")
  done
fi

docker rm -f solarwm-h3 2>/dev/null || true
docker run -d \
  --name solarwm-h3 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add IPC_LOCK \
  -e NODE_RANK="$NODE_RANK" \
  "${RDMA_ARGS[@]}" \
  -v /mnt/solar:/mnt/solar \
  -w /opt/SolarWM \
  "$IMAGE" sleep infinity
```

检查：

```bash
docker exec solarwm-h3 nvidia-smi
docker exec solarwm-h3 bash -lc 'ls -l /dev/infiniband || true'
docker exec solarwm-h3 bash -lc 'python -m solarwm environment probe'
```

生产环境不要使用 `--privileged`。如果 RDMA 设备无法通过，应修复 device
plugin/cgroup 配置，不要以 privileged 作为长期方案。

## 6. 下载代码、权重和数据

代码已经烘焙进镜像。构建用源码：

```bash
git clone https://github.com/Junchao-cs/SolarWM.git
cd SolarWM
git checkout <审核通过的 commit 或 release tag>
```

### 6.1 H3 权重

先在 Hugging Face 接受仓库条款，并确认第 0.1 节地域许可。仅在下载管理节点
注入短期 token：

```bash
python -m pip install --upgrade huggingface_hub
export HF_HOME=/mnt/solar/cache/hf
hf auth login

export SOLAR_MODEL_ROOT=/mnt/solar/models

hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" \
            "SolarWM-h3-33B-bid-stage0p5-158f/**" \
            "SolarWM-h3-33B-tf-stage1-158f/**" \
            "SolarWM-h3-33B-sgf-stage2-158f-fix/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

只做 Stage2 推理时，仅下载：

```bash
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" \
            "SolarWM-h3-33B-sgf-stage2-158f-fix/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

设置：

```bash
export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_STAGE0P5_INIT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f"
export H3_STAGE1_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-tf-stage1-158f"
export H3_STAGE2_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-sgf-stage2-158f-fix"
```

推荐 `-fix` Stage2 step-3900 EMA，不使用已被替代的原始 step-1200 包。

### 6.2 主数据仓库和 H3 latent generation

下载主仓库中的控制文件和索引：

```bash
export SOLAR_DATA_HOME=/mnt/solar/data/SolarWM-Data
export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"

hf download junchaoh-cs/SolarWM-Data \
  --repo-type dataset \
  --exclude "SolarWM-Data-Annotation/**" \
  --local-dir "$SOLAR_DATA_HOME"
```

H3 预编码训练数据不在上述主仓库中。它从 ModelScope International 单独发布：

https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1

按该仓库当前下载说明，把整个 generation 放到：

```text
/mnt/solar/data/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/
```

最终必须存在：

```bash
export H3_GENERATION="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1"
export H3_SUPPORT="$H3_GENERATION/support"

test -r "$H3_SUPPORT/h3_silence_153_158_170.safetensors"
test -r "$H3_SUPPORT/encoder_contract.json"
test -r "$SOLAR_DATA_ROOT/recipes/clean-158f-h3/latent-wds/minimax-h3-158f-768p-nomind-v1/train-index.jsonl.gz"
test -r "$SOLAR_DATA_ROOT/recipes/clean-158f-h3/latent-wds/minimax-h3-158f-768p-nomind-v1/test-index.jsonl.gz"
```

不要改 generation 目录名，不要把新打包的 tar 与另一分发版本的 index 混用。
SolarWM 会检查 shard 相对路径和声明的字节数。

### 6.3 Stage1 raw test camera

Stage1 配置的 `raw_test_index` 指向：

```text
recipes/clean-153f/raw-wds/test-index.jsonl.gz
```

Stage1 periodic validation 要读取这些 raw test shard 中的完整 camera trajectory。
只有 latent generation 不足以完成这一项。通过 SolarWM 数据访问申请或按
annotation package 重建对应 raw-WDS。未准备好时不要把完整 Stage1 run 标记为
可复现通过，也不要静默关闭 validation。

如果当前目标是 Stage2 推理或以发布的 Stage1 EMA 初始化 Stage2，可不自行运行
Stage1，从而不需要下载整个 raw 训练集。

## 7. 两节点公共环境

在共享盘创建 `/mnt/solar/h3-common.env`。该文件不包含 rank；`NODE_RANK`
已经在第 5 节创建容器时分别注入为 0 和 1：

```bash
export SOLAR_REPO=/opt/SolarWM
export SOLAR_MODEL_ROOT=/mnt/solar/models
export SOLAR_DATA_ROOT=/mnt/solar/data/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/mnt/solar/outputs

export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_STAGE0P5_INIT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f"
export H3_STAGE1_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-tf-stage1-158f"
export H3_STAGE2_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-sgf-stage2-158f-fix"
export H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"

export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=<node-0-private-ip>
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME=<actual-training-interface>
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=<actual-mlx5-list>
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
```

不要把 `NCCL_DEBUG=INFO` 永久用于长跑；通过预检后可改为 `WARN`，避免日志量
过大。不要设置 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1` 或强制某个
`NCCL_ALGO`，除非诊断证明有必要。

检查两节点路径内容一致：

```bash
source /mnt/solar/h3-common.env
test "$NODE_RANK" = 0 -o "$NODE_RANK" = 1
cd "$SOLAR_REPO"
test -r "$H3_BASE/transformer/config.json"
test -r "$H3_SUPPORT/encoder_contract.json"
python -m solarwm environment probe
solarwm config routes
```

## 8. 分层预检

### 8.1 单节点 GPU 和 FlashAttention

每台分别执行：

```bash
docker exec solarwm-h3 bash -lc '
python - <<PY
import torch
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8
for i in range(8):
    assert torch.cuda.get_device_capability(i) == (9, 0)
print([torch.cuda.get_device_name(i) for i in range(8)])

q = torch.randn(1, 1024, 16, 128, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
print(out.shape, torch.isfinite(out).all().item())
PY'
```

### 8.2 两节点 PyTorch/NCCL collective

把以下脚本保存到共享路径 `/mnt/solar/nccl_smoke.py`：

```python
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

x = torch.ones(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
for _ in range(5):
    dist.all_reduce(x)
torch.cuda.synchronize()

started = time.perf_counter()
for _ in range(20):
    dist.all_reduce(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started

if rank == 0:
    print({"world_size": dist.get_world_size(), "iterations": 20, "seconds": elapsed})
dist.barrier()
dist.destroy_process_group()
```

两节点同时运行，`NODE_RANK` 各不相同：

```bash
source /mnt/solar/h3-common.env
torchrun \
  --nnodes=2 \
  --node-rank="$NODE_RANK" \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id=solarwm-nccl-smoke \
  /mnt/solar/nccl_smoke.py
```

验收条件：

- 16 个 rank 全部初始化、完成 all-reduce 并正常退出；
- 日志中的网络 transport 是预期 IB/RoCE，而非意外 socket fallback；
- 无 timeout、NET/IB error、Xid；
- 连续运行三次稳定。

正式性能门槛应由云厂商按实例和网卡规格提供。不要只以“命令没报错”作为 RDMA
性能合格依据；同时运行云厂商推荐的 `nccl-tests all_reduce_perf`，与同规格
基线比较。

### 8.3 配置 resolve

Stage0.5 示例：

```bash
source /mnt/solar/h3-common.env
cd "$SOLAR_REPO"

solarwm config resolve \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=8 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/preflight-resolve"
```

人工复核输出中的：

- `family=minimax_h3`、stage/objective；
- world size、SP 和 global batch；
- 所有模型、数据、support 路径均无 `/path/to/...`；
- BF16、FSDP、LoRA rank；
- 初始化 checkpoint role 和 `weight_source`；
- 输出目录是新的目录。

### 8.4 最小代表性训练预检

正式长跑前，复制所选配置为 task-local preflight config，至少验证：

- 两节点 16 rank 启动；
- 一个完整 forward/backward/optimizer step；
- loss 和 gradient norm 为 finite；
- 一个 checkpoint transaction 完成并可重新加载；
- 所选 validation case 能读取并产出视频和 completion marker。

可用 `--set train.max_steps=2`、`--set checkpoint.save_every_steps=1` 和较小的
validation sample count 缩短预检，但不要把这种预检输出当正式 checkpoint。
Stage2 的一个 outer step 包含多次 critic update，仍会比 Stage0.5 显著更慢。

## 9. 正式训练启动

所有长跑都使用新的 `RUN_ID` 和新的共享输出目录。两节点必须使用完全相同的
`RUN_ID`、配置、路径和命令，只有 `NODE_RANK` 不同。

公共 launch 前缀：

```bash
source /mnt/solar/h3-common.env
cd "$SOLAR_REPO"
export RUN_ID=<唯一任务 ID，例如 h3-s05-20260923-01>
export RUN_DIR="$SOLAR_OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
```

### 9.1 Stage0.5

```bash
torchrun \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$RUN_ID" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=8 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$RUN_DIR" \
  2>&1 | tee -a "/mnt/solar/logs/${RUN_ID}-node${NODE_RANK}.log"
```

### 9.2 Stage1

确认 raw test camera shard 已准备好后运行：

```bash
torchrun \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$RUN_ID" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage1-158f-lora384-w6-sp2.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=8 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$RUN_DIR" \
  2>&1 | tee -a "/mnt/solar/logs/${RUN_ID}-node${NODE_RANK}.log"
```

### 9.3 Stage2

```bash
torchrun \
  --nnodes="$NNODES" \
  --node-rank="$NODE_RANK" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$RUN_ID" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage2-158f-lora384-w6-sp4.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=4 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE1_CHECKPOINT" \
  --set checkpoint.initialization.teacher.path="$H3_STAGE0P5_INIT" \
  --set checkpoint.initialization.critic.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$RUN_DIR" \
  2>&1 | tee -a "/mnt/solar/logs/${RUN_ID}-node${NODE_RANK}.log"
```

不要把初始化当 resume。初始化只加载指定角色权重并开始新优化；完整 resume
还要恢复 optimizer、scheduler、step、EMA、RNG 和 data-stream state。

## 10. 推理

### 10.1 两台节点并行做 158 帧 Stage2 推理

每台节点独立使用本机 8 GPU，分别设置唯一输出目录。此模式不需要跨节点
rendezvous：

```bash
source /mnt/solar/h3-common.env
cd "$SOLAR_REPO"
export INFER_ID=<唯一 ID，包含 node 名>

torchrun --standalone --nproc-per-node=8 \
  -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set checkpoint.weight_source=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/$INFER_ID"
```

如果 checkpoint 没有 EMA，才使用
`--set checkpoint.weight_source=live`。发布的 `-fix` 包应使用 EMA。

### 10.2 原始长度 Stage2 推理

按 `docs/backends/minimax-h3.md` 下载独立 test-set-v1，生成
`inference.plan`，使用：

```text
configs/examples/minimax_h3/infer-stage2-source-length-sp8.yaml
```

该路线每个任务占一台 8-GPU 节点。视频超过 960 帧时设置：

```bash
--set inference.stream_decode=true
```

## 11. 监控、checkpoint 和恢复

### 11.1 必须监控的信号

每个任务记录：

- image digest、SolarWM commit、resolved config hash；
- 16 个 rank 是否都存活；
- optimizer step、loss、gradient norm、step time；
- 每张 GPU 的利用率、显存、功耗、温度、ECC/Xid；
- host RAM、shared filesystem 容量和 inode；
- RDMA/NCCL error、网络吞吐；
- checkpoint 完成数；
- validation 的 pass/sample 数和失败 manifest。

辅助命令：

```bash
nvidia-smi dmon -s pucvmet -d 5
cat "/mnt/solar/logs/${RUN_ID}-node${NODE_RANK}.log"
df -h /mnt/solar/outputs
df -i /mnt/solar/outputs
```

SolarWM rank 0 会写：

```text
resolved-config.json
launch-manifest.json
run-result.json
```

只有看到完整 transaction/completion marker，且所有声明成员都可读，才把
checkpoint 标记为可恢复。不要从仍在写入的目录复制或 resume。

### 11.2 完整 resume

使用上一 run 的原始 `resolved-config.json` 和完整 checkpoint，写到新的输出
目录：

```bash
export PREVIOUS_RUN=/mnt/solar/outputs/<previous-run>
export RESUME_CKPT="$PREVIOUS_RUN/checkpoint_model_000200"
export RUN_ID=<new-resume-run-id>
export RUN_DIR="$SOLAR_OUTPUT_ROOT/$RUN_ID"

torchrun \
  --nnodes=2 \
  --node-rank="$NODE_RANK" \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$RUN_ID" \
  -m solarwm train \
  --config "$PREVIOUS_RUN/resolved-config.json" \
  --set checkpoint.resume_from="$RESUME_CKPT" \
  --set runtime.output_dir="$RUN_DIR"
```

先确认旧 checkpoint 在两个节点可见且 completion 状态完整。不要覆盖旧 run。

### 11.3 安全停止

优先向两个 `torchrun` launcher 发送 SIGTERM，并给 checkpoint/worker 合理退出
时间。只按精确 PID/容器和 `RUN_ID` 停止，不使用宽泛的 `pkill python`。

如果节点失联，先保留输出和日志。不要删除看似“不完整”的最后 checkpoint；
将它隔离标记为 incomplete，resume 选择上一个已完成 transaction。

## 12. BCS/Kubernetes 落地要点

在 BCS/Kubernetes 中，建议使用一个两副本、固定 rank 的 gang-scheduled Job
或支持 PyTorchJob 的 operator。关键配置：

- 每个 Pod request/limit `nvidia.com/gpu: 8`；
- 两个 Pod 必须落在不同的目标 H200 节点；
- 使用 nodeSelector/affinity 锁定 H200 141 GB 机型，禁止 MIG；
- 使用 gang scheduling，避免只启动一个 Pod 长时间占 8 GPU 等待另一个；
- `hostNetwork: true`，rendezvous 只绑定训练私网；
- `hostIPC: true` 或显式挂载足够大的 `/dev/shm`；
- `IPC_LOCK`、memlock unlimited；
- 通过 NVIDIA GPU Operator 暴露 GPU；
- 通过 NVIDIA Network Operator/RDMA device plugin 暴露正确 RDMA resource；
- ReadWriteMany PVC 以相同路径挂载 `/mnt/solar/outputs`；
- 模型/数据 PVC 只读挂载，token 使用 Secret，仅下载 Job 可见；
- Pod 0 提供稳定 DNS/IP，Pod ordinal 映射 `NODE_RANK`；
- termination grace period 足以让 launcher 退出，但不要依赖抢占时保存最后一步；
- 设置 PodDisruptionBudget，禁用训练节点自动缩容和随意 drain；
- 镜像使用 digest，不使用 `latest`。

如果使用普通 Deployment 而没有 gang scheduling、稳定 rank 和故障一致性，不适合
两机 torchrun 长训练。BCS 的 CNI、RDMA resource 名、RoCE GID/TC 和调度器名称
依集群实际配置填写，不能从别的集群照抄。

## 13. 常见故障定位

### 两机在 init_process_group 卡住

依次检查：

1. `NODE_RANK` 是否恰好为 0/1，`RUN_ID` 和 rendezvous endpoint 是否一致；
2. node-1 能否访问 node-0 私网 29500；
3. `NCCL_SOCKET_IFNAME` 是否是训练网卡；
4. 容器是否 host network；
5. RDMA device 是否进入容器；
6. 两节点 image digest、驱动和 NCCL 版本是否一致；
7. 是否意外走 socket fallback；
8. 云安全组、CNI、PFC/ECN、GID index 是否符合厂商文档。

先用第 8.2 节最小 collective 复现，不要直接反复启动 33B 模型。

### FlashAttention 报 no kernel image / undefined symbol

- 确认 H200 capability 是 9.0；
- 确认 FlashAttention 是在当前 torch 2.6.0 + CUDA 12.4 环境中编译；
- 确认安装时使用 `--no-build-isolation`；
- 清理旧 wheel/cache 后重建镜像；
- 不要从另一 torch/CUDA 镜像复制 `.so`。

### CUDA OOM

先检查是否有其他进程、MIG、僵尸 rank 和显存碎片。再核对是否使用正确配置：

- Stage0.5/1 SP2；
- Stage2 SP4；
- micro batch 1；
- activation checkpointing 开启；
- H3 33B base、student/teacher/critic role 未重复错误加载。

不要首先修改 latent geometry、关闭 FSDP 或混用其他 backend 配置。若标准两机
配置在空闲 H200 上仍 OOM，保存每 rank 峰值和 allocation trace，作为版本/加载
路径问题诊断。

### Stage1 在 smoke step 或 validation 失败

检查 `raw_test_index` 指向的每个 raw-WDS shard 是否存在且字节数匹配。只有
latent generation 时，Stage1 训练数据可以读取，但完整验证的 raw camera 仍会
缺失。

### checkpoint load 缺 key 或 shape mismatch

检查：

- base、Stage0.5、Stage1、Stage2 包是否属于 MiniMax-H3 33B；
- Stage2 student 使用 Stage1 EMA；
- teacher/critic 使用兼容 Stage0.5 EMA；
- 发布 Stage2 `-fix` 推理使用 `weight_source=ema`；
- checkpoint 目录是否保持完整，没有只复制某个 safetensors 文件。

### 数据读取慢或 rank 等待

- 检查共享文件系统吞吐/IOPS、metadata latency；
- 检查 WebDataset shard 是否跨 region；
- 把完整 generation 复制到同 region 的高吞吐盘或本地 NVMe；
- 保持两节点数据树和 index 一致；
- 不要让训练期间的下载任务抢占相同出口和文件系统；
- 使用 `solarwm data warm-cache` 时确保 cache 能容纳计划 working set。

### exit 137、无 Python traceback

优先看 Pod/容器 OOM、host RAM、`/dev/shm`、Kubernetes eviction 和云抢占事件，
而不是直接判定 CUDA OOM。

## 14. 上线验收清单

只有全部满足才进入正式长跑：

- H3 地域许可证已书面确认；
- 两节点各 8 张完整 H200，驱动/Fabric Manager 一致且健康；
- 镜像按 digest 固定，Python/PyTorch/CUDA/H3 包版本通过 probe；
- 节点内 NVLink/NVSwitch 健康；
- 跨节点 RDMA collective 连续三次通过且达到云厂商基线；
- 模型、数据、support、初始化 checkpoint 在两节点路径一致；
- Stage1 如启用，raw test camera shard 已准备；
- 配置 resolve 无 placeholder，world/SP/global batch 算术正确；
- 最小代表性训练完成 step、checkpoint reload 和 validation；
- 输出共享盘容量、inode、告警和备份策略就绪；
- 日志、GPU、主机、网络、checkpoint 监控已接入；
- 抢占、节点失联和 resume 流程至少演练一次；
- 正式 `RUN_ID`、负责人、镜像 digest、代码 commit、配置和数据 generation
  已登记。

## 15. 参考

- SolarWM H3：`docs/backends/minimax-h3.md`
- SolarWM 环境：`environments/README.md`
- SolarWM 数据访问：`docs/data-access.md`
- H3 latent generation：`docs/latent-wds.md`
- CUDA 12.4.1 release notes：
  https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/
- NVIDIA CUDA image：
  https://hub.docker.com/r/nvidia/cuda
- PyTorch 2.6.0 CUDA 12.4 image：
  https://hub.docker.com/r/pytorch/pytorch/tags?name=2.6.0-cuda12.4-cudnn9-devel
