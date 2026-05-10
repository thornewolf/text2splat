#!/usr/bin/env bash
set -euo pipefail

args=()
for arg in "$@"; do
  case "$arg" in
    --SiftExtraction.use_gpu)
      args+=(--FeatureExtraction.use_gpu)
      ;;
    --SiftMatching.use_gpu)
      args+=(--FeatureMatching.use_gpu)
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

exec colmap "${args[@]}"
