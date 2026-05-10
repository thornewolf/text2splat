#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_multiview_splat.sh -i /path/to/photos [options]

Runs a multi-image Gaussian splat workflow with Nerfstudio:
  photos -> ns-process-data images/COLMAP -> ns-train splatfacto -> export .ply -> render .mp4

Required:
  -i, --input-dir PATH        Directory containing object photos.
                             Not required when --load-config is provided.

Options:
  -w, --work-dir PATH         Working/output directory. Default: ./runs/multiview_splat
  -n, --run-name NAME         Run name. Default: basename(input-dir)-YYYYmmdd-HHMMSS
  --load-config PATH          Existing Nerfstudio config.yml. Skips processing/training.
  --method NAME               Nerfstudio method. Default: splatfacto
  --camera-type TYPE          Camera type for ns-process-data. Default: perspective
  --colmap-cmd PATH           COLMAP executable/wrapper. Default: scripts/colmap_compat.sh when present.
  --matching-method METHOD    COLMAP matching method. Default: exhaustive
  --process-gpu / --no-process-gpu
                             Use GPU for COLMAP feature extraction. Default: --process-gpu
  --max-num-iterations N      Optional training iteration cap.
  --device TYPE               Training device: cuda, mps, or cpu. Default: auto
  --install-env               Create/use ./.venv-nerfstudio and install nerfstudio with uv.
                             Requires colmap, ffmpeg, and pkg-config on macOS.
  --venv-dir PATH             Virtualenv path for --install-env. Default: ./.venv-nerfstudio
  --nerfstudio-source PATH     Local Nerfstudio clone for editable install.
                             Default: ../nerfstudio when it exists, otherwise PyPI.
  --python VERSION            Python version for --install-env. Default: 3.10
  --render-mode MODE          spiral, camera-path, or none. Default: spiral
  --camera-path PATH          Camera path JSON for --render-mode camera-path.
  --seconds N                 Spiral render seconds. Default: 6
  --frame-rate N              Render frame rate. Default: 24
  --radius N                  Spiral render radius. Default: 0.35
  --downscale-factor N        Render downscale factor. Default: 1.0
  --no-export                 Skip ns-export gaussian-splat.
  --dry-run                   Print commands without running them.
  -h, --help                  Show this help.

Examples:
  scripts/run_multiview_splat.sh -i /Users/thorn/Downloads/object_photos --install-env

  scripts/run_multiview_splat.sh \
    -i /Users/thorn/Downloads/object_photos \
    --max-num-iterations 7000 \
    --render-mode spiral

  scripts/run_multiview_splat.sh \
    -i /Users/thorn/Downloads/object_photos \
    --render-mode camera-path \
    --camera-path /Users/thorn/Downloads/camera_path.json
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

cuda_available() {
  python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

INPUT_DIR=""
LOAD_CONFIG=""
WORK_DIR="$(pwd)/runs/multiview_splat"
RUN_NAME=""
METHOD="splatfacto"
CAMERA_TYPE="perspective"
COLMAP_CMD=""
MATCHING_METHOD="exhaustive"
PROCESS_GPU="1"
MAX_NUM_ITERATIONS=""
DEVICE="auto"
INSTALL_ENV="0"
VENV_DIR="$(pwd)/.venv-nerfstudio"
NERFSTUDIO_SOURCE=""
PYTHON_VERSION="3.10"
RENDER_MODE="spiral"
CAMERA_PATH=""
SECONDS="6"
FRAME_RATE="24"
RADIUS="0.35"
DOWNSCALE_FACTOR="1.0"
EXPORT_SPLAT="1"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -w|--work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --load-config)
      LOAD_CONFIG="$2"
      shift 2
      ;;
    -n|--run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --camera-type)
      CAMERA_TYPE="$2"
      shift 2
      ;;
    --colmap-cmd)
      COLMAP_CMD="$2"
      shift 2
      ;;
    --matching-method)
      MATCHING_METHOD="$2"
      shift 2
      ;;
    --process-gpu)
      PROCESS_GPU="1"
      shift
      ;;
    --no-process-gpu)
      PROCESS_GPU="0"
      shift
      ;;
    --max-num-iterations)
      MAX_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --install-env)
      INSTALL_ENV="1"
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --nerfstudio-source)
      NERFSTUDIO_SOURCE="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --render-mode)
      RENDER_MODE="$2"
      shift 2
      ;;
    --camera-path)
      CAMERA_PATH="$2"
      shift 2
      ;;
    --seconds)
      SECONDS="$2"
      shift 2
      ;;
    --frame-rate)
      FRAME_RATE="$2"
      shift 2
      ;;
    --radius)
      RADIUS="$2"
      shift 2
      ;;
    --downscale-factor)
      DOWNSCALE_FACTOR="$2"
      shift 2
      ;;
    --no-export)
      EXPORT_SPLAT="0"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -z "$LOAD_CONFIG" ]]; then
  [[ -n "$INPUT_DIR" ]] || die "missing required --input-dir"
  [[ -d "$INPUT_DIR" ]] || die "input directory does not exist: $INPUT_DIR"
else
  [[ -f "$LOAD_CONFIG" ]] || die "config file does not exist: $LOAD_CONFIG"
fi

case "$RENDER_MODE" in
  spiral|camera-path|none) ;;
  *) die "--render-mode must be one of: spiral, camera-path, none" ;;
esac

if [[ "$RENDER_MODE" == "camera-path" ]]; then
  [[ -n "$CAMERA_PATH" ]] || die "--camera-path is required for --render-mode camera-path"
  [[ -f "$CAMERA_PATH" ]] || die "camera path JSON does not exist: $CAMERA_PATH"
fi

if [[ -z "$RUN_NAME" ]]; then
  if [[ -n "$INPUT_DIR" ]]; then
    RUN_NAME="$(basename "$INPUT_DIR")-$(date +%Y%m%d-%H%M%S)"
  else
    RUN_NAME="from-config-$(date +%Y%m%d-%H%M%S)"
  fi
fi

REPO_ROOT="$(repo_root)"
if [[ -z "$COLMAP_CMD" && -x "$REPO_ROOT/scripts/colmap_compat.sh" ]]; then
  COLMAP_CMD="$REPO_ROOT/scripts/colmap_compat.sh"
fi
mkdir -p "$(dirname "$WORK_DIR")"
WORK_DIR="$(cd "$(dirname "$WORK_DIR")" && pwd)/$(basename "$WORK_DIR")"
PROCESSED_DIR="$WORK_DIR/$RUN_NAME/processed"
TRAIN_OUTPUT_DIR="$WORK_DIR/$RUN_NAME/nerfstudio"
EXPORT_DIR="$WORK_DIR/$RUN_NAME/export"
RENDER_DIR="$WORK_DIR/$RUN_NAME/renders"
VIDEO_PATH="$RENDER_DIR/${RUN_NAME}.mp4"

if [[ -z "$LOAD_CONFIG" ]]; then
  IMAGE_COUNT="$(
    find "$INPUT_DIR" -type f \( \
      -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.heif' \
    \) | wc -l | tr -d ' '
  )"
  [[ "$IMAGE_COUNT" -gt 1 ]] || die "need at least 2 input images; found $IMAGE_COUNT"
else
  IMAGE_COUNT="0"
fi

mkdir -p "$PROCESSED_DIR" "$TRAIN_OUTPUT_DIR" "$EXPORT_DIR" "$RENDER_DIR"

if [[ "$INSTALL_ENV" == "1" ]]; then
  command_exists uv || die "uv is required for --install-env"
  if [[ ! -d "$VENV_DIR" ]]; then
    run uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  run uv pip install --upgrade pip setuptools wheel
  if [[ -z "$NERFSTUDIO_SOURCE" && -d "$REPO_ROOT/../nerfstudio/.git" ]]; then
    NERFSTUDIO_SOURCE="$REPO_ROOT/../nerfstudio"
  fi
  if [[ -n "$NERFSTUDIO_SOURCE" ]]; then
    run uv pip install -e "$NERFSTUDIO_SOURCE"
  else
    run uv pip install nerfstudio
  fi
elif [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

if [[ "$DRY_RUN" == "0" ]]; then
  if [[ -z "$LOAD_CONFIG" ]]; then
    command_exists colmap || die "COLMAP is required. On macOS: brew install colmap"
    command_exists ns-process-data || die "Nerfstudio CLI is missing. Re-run with --install-env or activate your Nerfstudio env."
    command_exists ns-train || die "Nerfstudio CLI is missing: ns-train"
  fi
  command_exists ffmpeg || die "FFmpeg is required for video rendering. On macOS: brew install ffmpeg"
  command_exists pkg-config || die "pkg-config is required to build PyAV. On macOS: brew install pkg-config"
  command_exists ns-render || die "Nerfstudio CLI is missing: ns-render"
  command_exists ns-export || die "Nerfstudio CLI is missing: ns-export"
  if [[ "$METHOD" == splatfacto* ]]; then
    if ! cuda_available; then
      die "Nerfstudio $METHOD currently requires CUDA for training. This machine has no CUDA device; use a CUDA Linux box for multi-image Gaussian splats."
    fi
  fi
else
  echo "Dry run: skipping executable checks and filesystem outputs created by Nerfstudio."
fi

if [[ -n "$LOAD_CONFIG" ]]; then
  echo "Config: $LOAD_CONFIG"
else
  echo "Input images: $IMAGE_COUNT"
fi
echo "Run name: $RUN_NAME"
echo "Work dir: $WORK_DIR/$RUN_NAME"

if [[ -n "$LOAD_CONFIG" ]]; then
  CONFIG_PATH="$LOAD_CONFIG"
else
  process_cmd=(
    ns-process-data images
    --data "$INPUT_DIR"
    --output-dir "$PROCESSED_DIR"
    --camera-type "$CAMERA_TYPE"
    --matching-method "$MATCHING_METHOD"
  )
  if [[ -n "$COLMAP_CMD" ]]; then
    process_cmd+=(--colmap-cmd "$COLMAP_CMD")
  fi
  if [[ "$PROCESS_GPU" == "1" ]]; then
    process_cmd+=(--gpu)
  else
    process_cmd+=(--no-gpu)
  fi
  run "${process_cmd[@]}"

train_cmd=(
    ns-train "$METHOD"
    --output-dir "$TRAIN_OUTPUT_DIR"
    --experiment-name "$RUN_NAME"
    --viewer.quit-on-train-completion True
  )
  if [[ "$DEVICE" != "auto" ]]; then
    train_cmd+=(--machine.device-type "$DEVICE")
  fi
  if [[ -n "$MAX_NUM_ITERATIONS" ]]; then
    train_cmd+=(--max-num-iterations "$MAX_NUM_ITERATIONS")
  fi
  train_cmd+=(--data "$PROCESSED_DIR")
  run "${train_cmd[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    CONFIG_PATH="$TRAIN_OUTPUT_DIR/$RUN_NAME/$METHOD/<timestamp>/config.yml"
  else
    CONFIG_PATH="$(find "$TRAIN_OUTPUT_DIR" -type f -name config.yml | sort | tail -n 1)"
    [[ -n "$CONFIG_PATH" ]] || die "could not find Nerfstudio config.yml under $TRAIN_OUTPUT_DIR"
  fi
fi

if [[ "$EXPORT_SPLAT" == "1" ]]; then
  export_cmd=(
    ns-export gaussian-splat
    --load-config "$CONFIG_PATH"
    --output-dir "$EXPORT_DIR"
    --output-filename "${RUN_NAME}.ply"
  )
  run "${export_cmd[@]}"
fi

case "$RENDER_MODE" in
  spiral)
    render_cmd=(
      ns-render spiral
      --load-config "$CONFIG_PATH"
      --output-path "$VIDEO_PATH"
      --seconds "$SECONDS"
      --frame-rate "$FRAME_RATE"
      --radius "$RADIUS"
      --downscale-factor "$DOWNSCALE_FACTOR"
    )
    run "${render_cmd[@]}"
    ;;
  camera-path)
    render_cmd=(
      ns-render camera-path
      --load-config "$CONFIG_PATH"
      --camera-path-filename "$CAMERA_PATH"
      --output-path "$VIDEO_PATH"
      --downscale-factor "$DOWNSCALE_FACTOR"
    )
    run "${render_cmd[@]}"
    ;;
  none)
    echo "Skipping render because --render-mode none was selected."
    ;;
esac

cat <<EOF

Done.
Config: $CONFIG_PATH
Export dir: $EXPORT_DIR
Render path: $VIDEO_PATH

To open the trained splat in Nerfstudio's viewer:
  ns-viewer --load-config "$CONFIG_PATH"
EOF
