#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="$ROOT_DIR/environments/h3-h200/Dockerfile"
BUILDER_NAME="${BUILDER_NAME:-solarwm-h3-builder}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"

if [[ -z "${IMAGE_REPOSITORY:-}" ]]; then
  cat >&2 <<'EOF'
ERROR: IMAGE_REPOSITORY is required.

Example:
  export IMAGE_REPOSITORY=my-tcr.tencentcloudcr.com/worldmodel/solarwm-h3
  ./environments/h3-h200/build-and-push.sh
EOF
  exit 2
fi

if [[ "$IMAGE_REPOSITORY" == *"://"* || "$IMAGE_REPOSITORY" == */ ]]; then
  echo "ERROR: IMAGE_REPOSITORY must not contain a URL scheme or trailing slash." >&2
  exit 2
fi

if [[ "$IMAGE_REPOSITORY" != */* ]]; then
  echo "ERROR: IMAGE_REPOSITORY must include registry host and repository path." >&2
  exit 2
fi

HOST_ARCH="$(uname -m)"
if [[ "$HOST_ARCH" != "x86_64" && "${ALLOW_CROSS_BUILD:-0}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: this H3 image should be built on an x86_64 Linux host.
Current host architecture: $HOST_ARCH

Move the SolarWM checkout to an x86_64 Linux builder or H200 node.
Cross-building FlashAttention under emulation is intentionally disabled.
EOF
  exit 2
fi

command -v docker >/dev/null || {
  echo "ERROR: docker is not installed." >&2
  exit 2
}
docker info >/dev/null
docker buildx version >/dev/null

SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORT_COMMIT="${SOURCE_COMMIT:0:12}"
SOURCE_URL="$(
  git -C "$ROOT_DIR" remote get-url origin 2>/dev/null \
    || echo https://github.com/Junchao-cs/SolarWM
)"

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null || true)" ]]; then
  DEFAULT_TAG="dev-${SHORT_COMMIT}-$(date -u +%Y%m%d%H%M%S)"
  echo "WARNING: working tree has uncommitted files; using development tag." >&2
else
  DEFAULT_TAG="${SHORT_COMMIT}-torch2.6-cu124"
fi

TAG="${TAG:-$DEFAULT_TAG}"
IMAGE="${IMAGE_REPOSITORY}:${TAG}"

if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create \
    --name "$BUILDER_NAME" \
    --driver docker-container \
    --use
else
  docker buildx use "$BUILDER_NAME"
fi
docker buildx inspect --bootstrap >/dev/null

cat <<EOF
Building and pushing:
  source:   $ROOT_DIR
  commit:   $SOURCE_COMMIT
  platform: linux/amd64
  flash jobs: $FLASH_ATTN_MAX_JOBS
  image:    $IMAGE
EOF

docker buildx build \
  --platform linux/amd64 \
  --pull \
  --push \
  --provenance=false \
  --sbom=false \
  --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" \
  --build-arg "SOURCE_URL=$SOURCE_URL" \
  --build-arg "FLASH_ATTN_MAX_JOBS=$FLASH_ATTN_MAX_JOBS" \
  --file "$DOCKERFILE" \
  --tag "$IMAGE" \
  "$ROOT_DIR"

INSPECT_OUTPUT="$(docker buildx imagetools inspect "$IMAGE")"
DIGEST="$(awk '$1 == "Digest:" {print $2; exit}' <<<"$INSPECT_OUTPUT")"

if [[ ! "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: image was pushed, but its registry digest could not be parsed." >&2
  echo "$INSPECT_OUTPUT" >&2
  exit 1
fi

IMMUTABLE_IMAGE="${IMAGE_REPOSITORY}@${DIGEST}"
OUTPUT_DIR="$ROOT_DIR/build"
OUTPUT_FILE="$OUTPUT_DIR/h3-image-reference.txt"
mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$IMMUTABLE_IMAGE" >"$OUTPUT_FILE"

cat <<EOF

SUCCESS

Pullable tag:
  $IMAGE

Immutable image:
  $IMMUTABLE_IMAGE

Saved to:
  $OUTPUT_FILE

Next:
  ./environments/h3-h200/verify-image.sh '$IMMUTABLE_IMAGE'
EOF
