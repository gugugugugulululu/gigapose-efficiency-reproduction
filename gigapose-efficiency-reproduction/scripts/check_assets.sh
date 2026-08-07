#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

WORKSPACE="${BOP_WORKSPACE:-}"
GIGAPOSE_REPO="${GIGAPOSE_REPO:-}"
CHECKPOINT="${GIGAPOSE_CHECKPOINT:-}"
PYTHON_BIN="${GP_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --gigapose-repo) GIGAPOSE_REPO="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$WORKSPACE" ]] || die "set --workspace or BOP_WORKSPACE"
[[ -n "$GIGAPOSE_REPO" ]] || GIGAPOSE_REPO="$WORKSPACE/code/gigapose"
[[ -n "$CHECKPOINT" ]] || CHECKPOINT="$WORKSPACE/pretrained/gigaPose_v1.ckpt"
PYTHON_BIN="$(resolve_python "$PYTHON_BIN")"

check_git_commit "$GIGAPOSE_REPO" "388e8bddd8a5443e284a7f70ad103d03f3f461c5"
check_sha256 "$CHECKPOINT" "0f60a23b03ddc41d2135c916ed1e66fb16f814f612dbde0305ae5a2c0f45c932"
require_file "$WORKSPACE/datasets/lmo/test_targets_bop19.json"
require_dir "$WORKSPACE/datasets/lmo/models_eval"

"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.executable)
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

log "asset checks passed"
