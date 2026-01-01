#!/usr/bin/env bash
set -euo pipefail

# --- pick Python 3.11 if available ---
PY_EXE="${PY_EXE:-python3.10}"
if ! command -v "$PY_EXE" >/dev/null 2>&1; then
  echo "$PY_EXE not found; falling back to python3"
  PY_EXE="python3"
fi

# --- create venv if missing ---
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment (.venv)..."
  "$PY_EXE" -m venv .venv
fi

# --- activate venv for this script run ---
# shellcheck disable=SC1091
source .venv/bin/activate


# --- tools ---
python -m pip install -U pip pip-tools

# --- detect whether we need to (re)compile ---
need_compile=false
if [ ! -f requirements.txt ] || [ requirements.in -nt requirements.txt ]; then
  need_compile=true
else
  # recompile if any included -r file is newer
  while IFS= read -r line; do
    case "$line" in
      -r\ *)
        inc="${line#-r }"
        inc="${inc%% #*}"; inc="${inc%%$'\r'}"
        if [ -f "$inc" ] && [ "$inc" -nt requirements.txt ]; then
          need_compile=true
        fi
        ;;
    esac
  done < requirements.in
fi

if $need_compile; then
  echo "Compiling requirements.txt from requirements.in..."
  pip-compile --quiet -o requirements.txt requirements.in
else
  echo "Requirements.txt is up to date."
fi

echo "Syncing environment to requirements.txt..."
pip-sync requirements.txt
echo "venv ready."

# --- If sourced, we're already active; if executed, open a subshell with venv active ---
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Opening a shell with .venv active. Type 'exit' to leave."
  tmp_rc="$(mktemp)"
  printf 'source "%s"\n' "$(pwd)/.venv/bin/activate" > "$tmp_rc"
  exec bash --noprofile --rcfile "$tmp_rc" -i
else
  echo "venv is active in this shell."
fi
