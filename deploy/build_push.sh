#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_push.sh <image-name> <tag> [registry-prefix]
# Example: ./build_push.sh mergekit-ga latest 123456789012.dkr.ecr.us-east-1.amazonaws.com

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <image-name> <tag> [registry-prefix]" >&2
  exit 1
fi

IMAGE_NAME="$1"
TAG="$2"
REGISTRY_PREFIX="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${REGISTRY_PREFIX}" ]]; then
  FULL_IMAGE_NAME="${REGISTRY_PREFIX}/${IMAGE_NAME}:${TAG}"
else
  FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
fi

# Build the container image from the repository root.
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ -n "${PLATFORM}" ]]; then
  echo "Building image for platform ${PLATFORM}. Override by exporting PLATFORM=<os/arch>."
  docker build \
    --platform "${PLATFORM}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${FULL_IMAGE_NAME}" \
    "${PROJECT_ROOT}"
else
  docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${FULL_IMAGE_NAME}" \
    "${PROJECT_ROOT}"
fi

echo "Built image ${FULL_IMAGE_NAME}."

if [[ -n "${REGISTRY_PREFIX}" ]]; then
  echo "Pushing image ${FULL_IMAGE_NAME} to registry. Ensure you are already authenticated."
  docker push "${FULL_IMAGE_NAME}"
fi
