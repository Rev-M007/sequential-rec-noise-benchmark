#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_environment.sh
#
# Creates (or reuses) a Python virtual environment in ./.venv, upgrades pip,
# installs requirements.txt, verifies the key imports and writes a fully pinned
# requirements-lock.txt for reproducibility.
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# Options (environment variables):
#   PYTHON=python3.11 ./setup_environment.sh   # force a specific interpreter
#   VENV_DIR=myenv    ./setup_environment.sh   # use a different venv folder
#
# Note: running the script creates the venv but cannot activate it in your
# current shell. Afterwards run:  source .venv/bin/activate
# ---------------------------------------------------------------------------
set -euo pipefail

# Always work relative to the project folder (the path contains spaces).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS="requirements.txt"
LOCK_FILE="requirements-lock.txt"

info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
fail()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. Pick a Python interpreter (3.9 - 3.12) -----------------------------
# PyTorch and Ray (a RecBole dependency) do not yet ship wheels for every new
# Python release, so newer interpreters are rejected rather than half-installed.
find_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        command -v "$PYTHON" || fail "PYTHON=$PYTHON not found on PATH."
        return
    fi
    for candidate in python3.12 python3.11 python3.10 python3 python3.9; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    fail "No Python 3 interpreter found. Install one, e.g. 'brew install python@3.11'."
}

[[ -f "$REQUIREMENTS" ]] || fail "$REQUIREMENTS not found in $SCRIPT_DIR."

PYTHON_BIN="$(find_python)"
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_MINOR="${PY_VERSION#*.}"
info "Using $PYTHON_BIN (Python $PY_VERSION)"

[[ "${PY_VERSION%%.*}" == "3" && "$PY_MINOR" -ge 9 && "$PY_MINOR" -le 12 ]] \
    || fail "Python $PY_VERSION is unsupported; use 3.9 - 3.12 (e.g. PYTHON=python3.11)."
[[ "$PY_MINOR" -ge 10 ]] \
    || warn "Python 3.9 works, but 3.10+ is recommended (brew install python@3.11)."

# ---- 2. Create the venv if it is missing or broken -------------------------
if [[ -f "$VENV_DIR/bin/activate" && -x "$VENV_DIR/bin/python" ]]; then
    info "Reusing existing virtual environment: $VENV_DIR"
else
    [[ -d "$VENV_DIR" ]] && { warn "$VENV_DIR exists but is incomplete; recreating it."; rm -rf "$VENV_DIR"; }
    info "Creating virtual environment: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# ---- 3. Activate it --------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated venv: $(python --version) at $(command -v python)"

# ---- 4. Upgrade packaging tools -------------------------------------------
info "Upgrading pip and wheel..."
# setuptools stays below 81: Ray (imported by RecBole's run_recbole) needs pkg_resources.
python -m pip install --upgrade pip wheel "setuptools<81"

# ---- 5. Install project dependencies --------------------------------------
info "Installing dependencies from $REQUIREMENTS (PyTorch is large; this can take several minutes)..."
python -m pip install -r "$REQUIREMENTS"

# ---- 6. Verify the installation -------------------------------------------
info "Verifying imports..."
python - <<'PY'
import importlib

packages = ["torch", "recbole", "numpy", "scipy", "pywt", "pandas",
            "matplotlib", "seaborn", "tqdm"]
for name in packages:
    module = importlib.import_module(name)
    print(f"  {name:<11} {getattr(module, '__version__', 'unknown')}")
importlib.import_module("recbole.quick_start")  # pulls in Ray; fails if pkg_resources is missing

import torch
if torch.cuda.is_available():
    device = f"CUDA ({torch.cuda.get_device_name(0)})"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "Apple MPS available (RecBole itself runs on CPU here)"
else:
    device = "CPU only"
print(f"  device      {device}")
PY

# ---- 7. Record exact versions ---------------------------------------------
python -m pip freeze > "$LOCK_FILE"
info "Exact package versions written to $LOCK_FILE"

info "Done. Activate the environment in your shell with:"
echo "    source \"$SCRIPT_DIR/$VENV_DIR/bin/activate\""
