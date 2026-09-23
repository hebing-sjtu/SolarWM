# 发布公开的 SolarWM H3 Runtime 镜像到 GHCR

公开镜像地址：

```text
ghcr.io/hebing-sjtu/solarwm-h3
```

该镜像只包含 SolarWM H3 runtime、CUDA/PyTorch 和 Python 依赖，不包含模型
权重、数据集或访问凭证。因此公开发布镜像不会把 H3 权重重新分发出去。

## 自动构建

GitHub workflow：

```text
.github/workflows/publish-h3-image.yml
```

它使用 GitHub 自动提供的 `GITHUB_TOKEN` 登录 GHCR，不需要创建或发送 PAT。
每次手工运行 workflow，或者推送 `h3-image-v*` tag 时，会构建 linux/amd64
镜像并发布以下 tag：

```text
ghcr.io/hebing-sjtu/solarwm-h3:h3-cu124-latest
ghcr.io/hebing-sjtu/solarwm-h3:h3-cu124-sha-<commit>
```

生产部署应在 workflow summary 中复制带 digest 的不可变地址：

```text
ghcr.io/hebing-sjtu/solarwm-h3@sha256:<digest>
```

## 首次发布后需要做的一次操作

个人账号下首次创建的 GHCR package 可能默认为 Private。首次 workflow 成功后：

1. 打开 `https://github.com/users/hebing-sjtu/packages/container/solarwm-h3/settings`。
2. 找到 `Danger Zone`。
3. 点击 `Change visibility`。
4. 选择 `Public` 并确认。

GitHub 提醒：公开后不能再改回私有。确认这个 package 只包含 runtime 后再操作。

公开完成后，任何人都可以无登录拉取：

```bash
docker pull ghcr.io/hebing-sjtu/solarwm-h3:h3-cu124-latest
```

推荐使用 digest：

```bash
docker pull ghcr.io/hebing-sjtu/solarwm-h3@sha256:<digest>
```

## H200 验证

```bash
export H3_IMAGE='ghcr.io/hebing-sjtu/solarwm-h3@sha256:<digest>'

EXPECTED_GPU_COUNT=8 \
  ./environments/h3-h200/verify-image.sh "$H3_IMAGE"
```

镜像默认执行 `sleep infinity`，可直接拉起：

```bash
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

## 仍需单独准备的运行资产

容器启动不等于 H3 模型已经可以推理。运行时还要挂载：

```text
/mnt/solar/models/SolarWM-h3-33B-base
/mnt/solar/models/SolarWM-h3-33B-sgf-stage2-158f-fix
/mnt/solar/data/SolarWM-Data/releases-v1
```

模型权重仍受 MiniMax H3 Community License 约束。公开 runtime 镜像不改变模型
在美国、欧盟、英国和韩国等排除地区的许可限制。
