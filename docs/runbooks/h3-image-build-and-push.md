# SolarWM H3 镜像构建与推送：傻瓜式教程

目标：执行一次脚本，最终得到下面这种可以直接拉取的不可变镜像地址：

```text
<TCR域名>/worldmodel/solarwm-h3@sha256:<64位摘要>
```

已经准备好的文件：

```text
environments/h3-h200/Dockerfile
environments/h3-h200/Dockerfile.dockerignore
environments/h3-h200/build-and-push.sh
environments/h3-h200/verify-image.sh
```

## 一句话结论

- 在哪里构建：同地域的 x86_64 Ubuntu Linux 构建机。最省事的是临时使用一台
  H200 节点的宿主机；构建本身不占 GPU。
- 基础容器：`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`。
- 推送到哪里：BCS 集群同地域的私有 TCR Enterprise 仓库。
- 不建议：在 Apple Silicon Mac 上跨架构编译 FlashAttention，或推到公共
  Docker Hub。
- 镜像包含：SolarWM 代码、Python 3.10、PyTorch 2.6.0、CUDA 12.4 runtime、
  FlashAttention 2.8.3 和 H3 Python 依赖。
- 镜像不包含：H3 权重、数据集、HF/ModelScope token。它们在运行时挂载。

## 第 1 步：在 TCR 创建私有仓库

在腾讯云控制台完成：

1. 打开“容器镜像服务 TCR”。
2. 选择与海外 BCS/H200 集群相同的 region。
3. 优先使用企业版实例。
4. 创建私有 namespace：`worldmodel`。
5. 创建私有 repository：`solarwm-h3`。
6. 配置该 BCS 集群访问 TCR；优先使用控制台提供的内网访问链路。
7. 创建一个用于推送的临时访问凭证。

从 TCR 页面抄下三个值：

```bash
export REGISTRY_HOST='<TCR访问域名，不带 https://>'
export TCR_USERNAME='<TCR用户名>'
export IMAGE_REPOSITORY="$REGISTRY_HOST/worldmodel/solarwm-h3"
```

`REGISTRY_HOST` 示例仅用于理解格式：

```text
example.tencentcloudcr.com
```

必须以你自己的 TCR 控制台显示值为准。不要把镜像推到另一个国家/region 的
registry，否则每次拉取慢且产生跨区流量。

如果组织已经有同地域私有 Harbor，也可以把 `IMAGE_REPOSITORY` 设置为：

```text
harbor.example.com/worldmodel/solarwm-h3
```

后续命令完全相同。

## 第 2 步：准备一台构建机

推荐配置：

- x86_64 Ubuntu 22.04；
- 至少 16 CPU、32 GB RAM；
- 至少 100 GB 可用磁盘；
- 能访问 Docker Hub、PyTorch wheel index 和 PyPI；
- 能访问目标 TCR；
- 安装 Docker Engine 和 buildx。

最简单的选择就是登录一台海外 H200 节点宿主机。不要在普通 BCS 业务 Pod 里
挂 Docker socket 构建镜像。

检查：

```bash
uname -m
docker version
docker buildx version
df -h
free -h
```

`uname -m` 必须输出：

```text
x86_64
```

如果 Docker 未安装，按 Docker 官方 Ubuntu 文档安装：
https://docs.docker.com/engine/install/ubuntu/

构建阶段不要求 NVIDIA Container Toolkit；最后在 H200 上验证镜像时才要求
`docker run --gpus all` 可用。

## 第 3 步：把当前 SolarWM 代码放到构建机

如果当前代码只在本机 Mac，最直接的方法是在 Mac 终端执行：

```bash
rsync -az \
  --exclude 'build/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  /Users/binghe/project/fastvideo/SolarWM/ \
  <构建机用户>@<构建机IP>:/opt/build/SolarWM/
```

然后登录构建机：

```bash
ssh <构建机用户>@<构建机IP>
cd /opt/build/SolarWM
ls environments/h3-h200
```

应该看到：

```text
Dockerfile
Dockerfile.dockerignore
build-and-push.sh
verify-image.sh
```

如果代码已经提交到一个构建机能访问的 Git 仓库，也可以直接 clone 对应 commit。
不要 clone 上游仓库后假设它已经包含本教程新增的构建脚本。

## 第 4 步：登录 TCR

在构建机执行：

```bash
export REGISTRY_HOST='<TCR访问域名，不带 https://>'
export TCR_USERNAME='<TCR用户名>'
export IMAGE_REPOSITORY="$REGISTRY_HOST/worldmodel/solarwm-h3"

read -rsp '请输入 TCR 临时密码: ' TCR_PASSWORD
echo
printf '%s' "$TCR_PASSWORD" | \
  docker login "$REGISTRY_HOST" \
    --username "$TCR_USERNAME" \
    --password-stdin
unset TCR_PASSWORD
```

看到 `Login Succeeded` 才继续。

常见错误：

- 不要在 `REGISTRY_HOST` 前写 `https://`；
- 不要把 namespace/repository 填进 `REGISTRY_HOST`；
- `IMAGE_REPOSITORY` 才是完整的仓库路径；
- 不要把密码写进脚本、Dockerfile、Git 或聊天记录。

## 第 5 步：一条命令构建并推送

在 SolarWM 根目录执行：

```bash
cd /opt/build/SolarWM
chmod +x environments/h3-h200/*.sh

export IMAGE_REPOSITORY="$REGISTRY_HOST/worldmodel/solarwm-h3"
./environments/h3-h200/build-and-push.sh
```

脚本会自动完成：

1. 检查当前机器是否为 x86_64；
2. 创建 buildx builder；
3. 拉取 CUDA 12.4.1 + cuDNN 开发镜像；
4. 安装 Python 3.10；
5. 安装 PyTorch 2.6.0 + CUDA 12.4 wheel；
6. 安装 H3 固定版本依赖；
7. 为 H100/H200 的 SM90 编译 FlashAttention 2.8.3；
8. 安装当前 SolarWM 源码；
9. 执行 Decord/H3 依赖 import 和 SolarWM environment probe；
10. 推送到 TCR；
11. 查询 registry digest；
12. 把最终不可变地址写入：
    `build/h3-image-reference.txt`。

第一次构建可能需要较长时间，主要耗时在拉取 PyTorch/CUDA 和编译
FlashAttention。不要中途关闭 SSH。推荐在 `tmux` 中运行：

```bash
tmux new -s solarwm-image
cd /opt/build/SolarWM
export IMAGE_REPOSITORY="$REGISTRY_HOST/worldmodel/solarwm-h3"
./environments/h3-h200/build-and-push.sh 2>&1 | \
  tee build/h3-image-build.log
```

成功后会显示：

```text
SUCCESS

Pullable tag:
  example.tencentcloudcr.com/worldmodel/solarwm-h3:<tag>

Immutable image:
  example.tencentcloudcr.com/worldmodel/solarwm-h3@sha256:<digest>
```

读取最终地址：

```bash
export H3_IMAGE="$(cat build/h3-image-reference.txt)"
echo "$H3_IMAGE"
```

正式 Job 永远使用 `H3_IMAGE` 这个带 `@sha256:` 的地址，不使用浮动 tag。

## 第 6 步：在一台 H200 节点验证镜像

确认该 H200 节点已安装 NVIDIA Container Toolkit，然后将
`build/h3-image-reference.txt` 中的地址复制过去：

```bash
export H3_IMAGE='<TCR域名>/worldmodel/solarwm-h3@sha256:<digest>'
docker login "$REGISTRY_HOST"

EXPECTED_GPU_COUNT=8 \
  ./environments/h3-h200/verify-image.sh "$H3_IMAGE"
```

脚本会检查：

- 容器能看到 8 张 GPU；
- GPU compute capability 为 9.0；
- Python、PyTorch、CUDA、Diffusers、Transformers、PEFT 和 FlashAttention
  版本正确；
- Decord 和 H3 关键依赖可以正常 import；
- SolarWM environment probe 通过；
- FlashAttention 在真实 H200 上执行并返回 finite 结果。

最终应看到：

```text
IMAGE VERIFICATION PASSED
```

如果只是用一张 GPU 的临时节点做镜像检查：

```bash
EXPECTED_GPU_COUNT=1 \
  ./environments/h3-h200/verify-image.sh "$H3_IMAGE"
```

## 第 7 步：直接拉起一个容器

在 H200 节点执行：

```bash
export H3_IMAGE='<TCR域名>/worldmodel/solarwm-h3@sha256:<digest>'

docker pull "$H3_IMAGE"
docker rm -f solarwm-h3 2>/dev/null || true

docker run -d \
  --name solarwm-h3 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add IPC_LOCK \
  -v /mnt/solar:/mnt/solar \
  -w /opt/SolarWM \
  "$H3_IMAGE"
```

检查：

```bash
docker ps --filter name=solarwm-h3
docker exec solarwm-h3 nvidia-smi
docker exec solarwm-h3 python -m solarwm environment probe
docker exec solarwm-h3 bash -lc 'solarwm config routes'
```

进入容器：

```bash
docker exec -it solarwm-h3 bash
```

镜像默认执行 `sleep infinity`，所以无需额外 command 就能保持运行。

注意：这一步证明镜像可以拉取、启动并访问 GPU，不代表已经能执行 H3 推理。
实际推理还需要把下面路径准备并挂载到 `/mnt/solar`：

```text
/mnt/solar/models/SolarWM-h3-33B-base
/mnt/solar/models/SolarWM-h3-33B-sgf-stage2-158f-fix
/mnt/solar/data/SolarWM-Data/releases-v1
```

权重和数据准备见：

```text
docs/runbooks/h3-2x8-h200-overseas.md
```

## 第 8 步：让 BCS 可以拉取镜像

首选方法是在 TCR 控制台把企业版实例和目标 BCS 集群绑定，使用平台托管的
registry credential。

如果平台要求 `imagePullSecret`，建议在一个隔离的临时 Docker config 中登录，
再创建 Secret：

```bash
export NAMESPACE='<BCS namespace>'
export REGISTRY_HOST='<TCR访问域名>'
export TCR_USERNAME='<TCR用户名>'

export DOCKER_CONFIG="$(mktemp -d)"
read -rsp '请输入 TCR 拉取密码: ' TCR_PASSWORD
echo
printf '%s' "$TCR_PASSWORD" | \
  docker login "$REGISTRY_HOST" \
    --username "$TCR_USERNAME" \
    --password-stdin
unset TCR_PASSWORD

kubectl -n "$NAMESPACE" create secret generic tcr-pull \
  --type=kubernetes.io/dockerconfigjson \
  --from-file=.dockerconfigjson="$DOCKER_CONFIG/config.json" \
  --dry-run=client -o yaml | kubectl apply -f -

rm -rf "$DOCKER_CONFIG"
unset DOCKER_CONFIG
```

工作负载中使用：

```yaml
spec:
  imagePullSecrets:
    - name: tcr-pull
  containers:
    - name: solarwm
      image: <TCR域名>/worldmodel/solarwm-h3@sha256:<digest>
```

如果 TCR 已与 BCS 集群集成，通常不需要手工创建 Secret，以平台实际配置为准。

## 第 9 步：最小 BCS 拉起检查

把以下内容中的 namespace、image、GPU resource 名和 H200 node label 换成集群
实际值：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: solarwm-h3-image-check
  namespace: <namespace>
spec:
  restartPolicy: Never
  imagePullSecrets:
    - name: tcr-pull
  nodeSelector:
    <集群的H200节点标签key>: <集群的H200节点标签value>
  containers:
    - name: solarwm
      image: <TCR域名>/worldmodel/solarwm-h3@sha256:<digest>
      imagePullPolicy: IfNotPresent
      command: ["bash", "-lc"]
      args:
        - |
          python -m solarwm environment probe
          nvidia-smi
          sleep infinity
      resources:
        limits:
          nvidia.com/gpu: "1"
      volumeMounts:
        - name: shm
          mountPath: /dev/shm
  volumes:
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 64Gi
```

执行：

```bash
kubectl apply -f solarwm-h3-image-check.yaml
kubectl -n <namespace> get pod solarwm-h3-image-check -w
kubectl -n <namespace> logs solarwm-h3-image-check
kubectl -n <namespace> exec -it solarwm-h3-image-check -- nvidia-smi
```

完成后删除测试 Pod：

```bash
kubectl -n <namespace> delete pod solarwm-h3-image-check
```

## 出错时先看这里

### `exec format error`

镜像被错误构建为 arm64。回到 x86_64 Linux 构建机重新执行脚本。不要在 M 系列
Mac 上直接构建正式镜像。

### `unauthorized` 或 `denied`

检查：

- `docker login` 使用的域名是否与 `IMAGE_REPOSITORY` 第一段完全一致；
- namespace/repository 是否存在；
- 当前凭证是否有 push 权限；
- BCS 使用的 Secret 是否有 pull 权限；
- TCR 是否允许当前 VPC/公网来源访问。

### FlashAttention 编译时构建机 OOM

降低并行度后重建：

```bash
# 修改 Dockerfile 中 MAX_JOBS=8 为 4，或给构建机增加 RAM。
```

不要改用来源不明的预编译 `.so`。

### H200 上显示 `no kernel image is available`

确认镜像是由本教程 Dockerfile 构建，且：

```text
TORCH_CUDA_ARCH_LIST=9.0
```

重新构建，不要复用其他 CUDA/PyTorch 环境生成的 FlashAttention wheel。

### BCS 为 `ImagePullBackOff`

查看：

```bash
kubectl -n <namespace> describe pod <pod-name>
```

重点看 registry DNS、TCR 网络访问、image digest、Secret 名称和 Secret 所在
namespace。Secret 不能跨 namespace 使用。

## 最终应该保留的交付物

```text
镜像 tag：
镜像 digest 地址：
TCR region：
TCR repository：
SolarWM commit：
构建日志：
verify-image.sh 结果：
BCS image-check Pod 结果：
```

其中生产部署真正使用的是：

```text
<TCR域名>/worldmodel/solarwm-h3@sha256:<digest>
```
