#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

resolve_python() {
  local requested="${1:-}"
  if [[ -n "$requested" ]]; then
    [[ -x "$requested" ]] || die "Python is not executable: $requested"
    printf '%s\n' "$requested"
  elif [[ -n "${GP_PYTHON:-}" && -x "${GP_PYTHON}" ]]; then
    printf '%s\n' "$GP_PYTHON"
  else
    command -v python3 || die "python3 not found"
  fi
}

source_optional_env() {
  local path="${1:-}"
  if [[ -n "$path" ]]; then
    require_file "$path"
    # shellcheck disable=SC1090
    source "$path"
  fi
}

check_git_commit() {
  local repo="$1"
  local expected="$2"
  require_dir "$repo/.git"
  local actual
  actual="$(git -C "$repo" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || die "GigaPose commit mismatch: expected $expected, got $actual"
}

check_sha256() {
  local path="$1"
  local expected="$2"
  require_file "$path"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "SHA256 mismatch for $path: expected $expected, got $actual"
}

find_latest_refined_csv() {
  local directory="$1"
  find "$directory" -type f -name '*.csv' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}
