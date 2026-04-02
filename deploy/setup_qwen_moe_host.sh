#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-qwen_moe_host}"
INSTALL_MODE="${INSTALL_MODE:-env}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment_qwen_moe_host.yml}"
CONDA_LOCK_FILE="${CONDA_LOCK_FILE:-$ROOT_DIR/conda_explicit_qwen_moe_host.txt}"
PIP_LOCK_FILE="${PIP_LOCK_FILE:-$ROOT_DIR/requirements_qwen_moe_host.txt}"
CORE_REQ_FILE="${CORE_REQ_FILE:-$ROOT_DIR/requirements_inference_core.txt}"
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
    env|yaml|portable)
        require_file "$ENV_FILE"
        echo "Creating env '$ENV_NAME' from: $ENV_FILE"
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
        ;;
    lock|exact)
        require_file "$CONDA_LOCK_FILE"
        require_file "$PIP_LOCK_FILE"
        echo "Creating env '$ENV_NAME' from exact conda lock: $CONDA_LOCK_FILE"
        conda create -n "$ENV_NAME" --file "$CONDA_LOCK_FILE" -y
        echo "Installing exact pip lock: $PIP_LOCK_FILE"
        conda run -n "$ENV_NAME" python -m pip install -r "$PIP_LOCK_FILE"
        ;;
    core|minimal)
        require_file "$CORE_REQ_FILE"
        echo "Creating minimal inference env '$ENV_NAME'"
        conda create -n "$ENV_NAME" python=3.10 pip -y
        echo "Installing curated inference requirements: $CORE_REQ_FILE"
        conda run -n "$ENV_NAME" python -m pip install -r "$CORE_REQ_FILE"
        ;;
    *)
        echo "Unsupported INSTALL_MODE=$INSTALL_MODE" >&2
        echo "Expected one of: env, lock, core" >&2
        exit 1
        ;;
esac

conda activate "$ENV_NAME"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo
echo "Created env: $ENV_NAME"
echo "Install mode: $INSTALL_MODE"
python --version
pip --version
echo
python "$ROOT_DIR/check_inference_env.py"
echo
echo "Activate later with:"
echo "  source \"$CONDA_HOME/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_NAME"
echo
echo "Guide: $ROOT_DIR/README.md"
