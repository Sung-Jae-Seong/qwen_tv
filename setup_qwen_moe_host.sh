#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/milab/miniconda3}"
ENV_NAME="${ENV_NAME:-qwen_moe_host}"
INSTALL_MODE="${INSTALL_MODE:-conda_env}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment_qwen_moe_host.yml}"
CONDA_LOCK_FILE="${CONDA_LOCK_FILE:-$ROOT_DIR/conda_explicit_qwen_moe_host.txt}"
PIP_LOCK_FILE="${PIP_LOCK_FILE:-$ROOT_DIR/requirements_qwen_moe_host.txt}"
RECREATE="${RECREATE:-0}"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
}

if [[ ! -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under CONDA_HOME=$CONDA_HOME" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    if [[ "$RECREATE" == "1" ]]; then
        echo "Removing existing env: $ENV_NAME"
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Environment already exists: $ENV_NAME"
        echo "Set RECREATE=1 to rebuild it."
        exit 0
    fi
fi

case "$INSTALL_MODE" in
    conda_env|env|yaml|portable)
        require_file "$ENV_FILE"
        echo "Creating host env '$ENV_NAME' from conda env file: $ENV_FILE"
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
        ;;
    lock|exact)
        require_file "$CONDA_LOCK_FILE"
        require_file "$PIP_LOCK_FILE"
        echo "Creating host env '$ENV_NAME' from raw conda lock: $CONDA_LOCK_FILE"
        conda create -n "$ENV_NAME" --file "$CONDA_LOCK_FILE" -y
        echo "Installing pinned pip packages: $PIP_LOCK_FILE"
        conda run -n "$ENV_NAME" python -m pip install -r "$PIP_LOCK_FILE"
        ;;
    *)
        echo "Unsupported INSTALL_MODE=$INSTALL_MODE (expected: conda_env or lock)" >&2
        exit 1
        ;;
esac

conda activate "$ENV_NAME"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Created env: $ENV_NAME"
echo "Install mode: $INSTALL_MODE"
python --version
pip --version
python - <<'PY'
import sys

packages = {}
for name in ["torch", "transformers", "vllm", "numpy", "pandas"]:
    try:
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        packages[name] = f"ERROR: {exc}"

print("python_runtime", sys.version.split()[0])
for name in ["torch", "transformers", "vllm", "numpy", "pandas"]:
    print(f"{name} {packages[name]}")

try:
    import torch
except Exception:
    torch = None

if torch is not None:
    print("cuda_available", torch.cuda.is_available())
    print("visible_device_count", torch.cuda.device_count())
else:
    print("cuda_available", "ERROR")
    print("visible_device_count", "ERROR")
PY
echo
echo "If you need the current shell to use this env:"
echo "  source \"$CONDA_HOME/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_NAME"
echo
echo "Conda env file: $ENV_FILE"
echo "Raw conda lock: $CONDA_LOCK_FILE"
echo "Pinned pip    : $PIP_LOCK_FILE"
echo "Refresh files : $ROOT_DIR/export_qwen_moe_host_env.sh"
echo "Env guide     : $ROOT_DIR/QWEN_MOE_HOST_ENV_SETUP.md"
