#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_environment.sh
#
# Installs project dependencies in a Colab-safe order:
#   1. torch / torchvision / torchaudio (CUDA wheel already present on Colab)
#   2. recbole==1.2.1 with --no-deps  (avoids Ray / dependency-resolver crashes)
#   3. the small set of packages RecBole actually needs at import time
#
# Works both in Google Colab (system Python) and locally (optional .venv).
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# Options:
#   SKIP_VENV=1 ./setup_environment.sh   # force system/Colab Python (no venv)
#   PYTHON=python3 ./setup_environment.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
SKIP_VENV="${SKIP_VENV:-0}"

info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
fail()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# Colab sets this; also honour an explicit SKIP_VENV=1.
IN_COLAB=0
if [[ -n "${COLAB_RELEASE_TAG:-}" || -d /content ]] && python -c "import google.colab" 2>/dev/null; then
    IN_COLAB=1
    SKIP_VENV=1
fi

find_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        command -v "$PYTHON" || fail "PYTHON=$PYTHON not found on PATH."
        return
    fi
    for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    fail "No Python 3 interpreter found."
}

PYTHON_BIN="$(find_python)"
info "Using $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])'))"

# ---- Optional local venv (skipped on Colab) --------------------------------
if [[ "$SKIP_VENV" != "1" ]]; then
    if [[ -f "$VENV_DIR/bin/activate" && -x "$VENV_DIR/bin/python" ]]; then
        info "Reusing existing virtual environment: $VENV_DIR"
    else
        [[ -d "$VENV_DIR" ]] && { warn "$VENV_DIR incomplete; recreating."; rm -rf "$VENV_DIR"; }
        info "Creating virtual environment: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$(command -v python)"
    info "Activated venv: $($PYTHON_BIN --version) at $PYTHON_BIN"
else
    info "Skipping venv (Colab / SKIP_VENV=1); installing into current interpreter."
fi

info "Upgrading pip / wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip wheel

# ---- Colab-safe install order (avoids Ray dependency resolution) -----------
info "Installing PyTorch..."
"$PYTHON_BIN" -m pip install torch torchvision torchaudio

info "Installing RecBole 1.2.1 without transitive deps (avoids Ray)..."
"$PYTHON_BIN" -m pip install "recbole==1.2.1" --no-deps

info "Installing RecBole runtime deps + project packages..."
"$PYTHON_BIN" -m pip install \
    tensorboardX colorlog colorama thop texttable \
    "numpy>=1.23.0" "pandas>=1.5.0" scipy pywavelets \
    matplotlib seaborn tqdm

# ---- Verify ---------------------------------------------------------------
info "Verifying imports..."
"$PYTHON_BIN" - <<'PY'
import importlib
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Apply the same lightweight stubs train_model.py uses, so verification
# succeeds even when Ray is absent (expected with --no-deps).
import sys, types
if "ray" not in sys.modules:
    ray = types.ModuleType("ray")
    tune = types.ModuleType("tune")
    ray.tune = tune
    sys.modules["ray"] = ray
    sys.modules["ray.tune"] = tune

packages = [
    "torch", "recbole", "numpy", "scipy", "pywt", "pandas",
    "matplotlib", "seaborn", "tqdm", "tensorboardX",
]
for name in packages:
    module = importlib.import_module(name)
    print(f"  {name:<12} {getattr(module, '__version__', 'ok')}")

import torch
if torch.cuda.is_available():
    device = f"CUDA ({torch.cuda.get_device_name(0)})"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "Apple MPS"
else:
    device = "CPU only"
print(f"  device       {device}")
PY

"$PYTHON_BIN" -m pip freeze > requirements-lock.txt
info "Pinned versions written to requirements-lock.txt"

if [[ "$SKIP_VENV" != "1" ]]; then
    info "Done. Activate with:  source \"$SCRIPT_DIR/$VENV_DIR/bin/activate\""
else
    info "Done. Packages installed into the current Python environment."
fi
