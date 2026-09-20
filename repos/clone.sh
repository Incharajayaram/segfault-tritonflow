#!/usr/bin/env bash
# Shallow-clone reference repositories for the TritonFlow project.
set -u
cd "$(dirname "$0")"

clone() {
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[skip] $dir already present"
    return
  fi
  echo "[clone] $url -> $dir"
  git clone --depth 1 --no-tags --single-branch "$url" "$dir" 2>&1 | tail -3
  echo "[done ] $dir ($(du -sh "$dir" 2>/dev/null | cut -f1))"
}

clone https://github.com/microsoft/triton-shared.git triton-shared
clone https://github.com/Cambricon/triton-linalg.git triton-linalg
clone https://github.com/intel/intel-xpu-backend-for-triton.git intel-xpu-backend-for-triton
clone https://github.com/pytorch/pytorch.git pytorch

echo "=== fetching pytorch PR #165958 ==="
if [ -d pytorch/.git ]; then
  ( cd pytorch && git fetch --depth 1 origin pull/165958/head:pr-165958 2>&1 | tail -3 )
  echo "[done ] PR ref pr-165958"
fi

echo "=== disk usage ==="
du -sh triton-shared triton-linalg intel-xpu-backend-for-triton pytorch 2>/dev/null
echo "=== CLONE SCRIPT COMPLETE ==="
