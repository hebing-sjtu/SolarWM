#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${1:-}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"

if [[ -z "$IMAGE" ]]; then
  cat >&2 <<'EOF'
Usage:
  ./environments/h3-h200/verify-image.sh \
    registry.example.com/worldmodel/solarwm-h3@sha256:...
EOF
  exit 2
fi

command -v docker >/dev/null || {
  echo "ERROR: docker is not installed." >&2
  exit 2
}

docker pull "$IMAGE"
docker image inspect "$IMAGE" \
  --format 'image={{.Id}} architecture={{.Architecture}} labels={{json .Config.Labels}}'

docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -e "EXPECTED_GPU_COUNT=$EXPECTED_GPU_COUNT" \
  "$IMAGE" \
  bash -lc '
set -Eeuo pipefail
python --version
python -m solarwm environment probe
python - <<'"'"'PY'"'"'
import os

import decord
import diffusers
import flash_attn
import peft
import torch
import transformers
from flash_attn import flash_attn_func

expected_gpu_count = int(os.environ["EXPECTED_GPU_COUNT"])
assert torch.cuda.is_available(), "CUDA is not available inside the container"
assert torch.cuda.device_count() == expected_gpu_count, (
    torch.cuda.device_count(),
    expected_gpu_count,
)
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert decord.__version__ == "0.6.0", decord.__version__
assert diffusers.__version__ == "0.40.0", diffusers.__version__
assert transformers.__version__ == "5.12.1", transformers.__version__
assert peft.__version__ == "0.20.0", peft.__version__
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__

for index in range(expected_gpu_count):
    capability = torch.cuda.get_device_capability(index)
    assert capability == (9, 0), (index, capability)

q = torch.randn(
    1,
    1024,
    16,
    128,
    device="cuda",
    dtype=torch.bfloat16,
)
output = flash_attn_func(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(output).all()

print(
    {
        "status": "PASS",
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "flash_attention_output": tuple(output.shape),
    }
)
PY
'

cat <<EOF

IMAGE VERIFICATION PASSED

Use this image by digest:
  $IMAGE
EOF
