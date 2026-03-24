#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/milab/miniconda3}"
ENV_NAME="${ENV_NAME:-qwen_test}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment_qwen_test.yml}"
RECREATE="${RECREATE:-0}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Environment file not found: $ENV_FILE" >&2
    exit 1
fi

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

echo "Creating env '$ENV_NAME' from $ENV_FILE"
conda env create -n "$ENV_NAME" -f "$ENV_FILE"

conda activate "$ENV_NAME"
echo "Created env: $ENV_NAME"
python --version
python - <<'PY'
import sys
import torch
import transformers
import pandas
import huggingface_hub

print("python_runtime", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("cuda_available", torch.cuda.is_available())
print("visible_device_count", torch.cuda.device_count())
PY
