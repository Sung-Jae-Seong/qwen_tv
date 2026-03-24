# qwen_moe_host Environment Reproduction

`qwen_moe_host` is the environment currently used to run Qwen3.5 9B / 27B vLLM inference in this repo.

## Environment fingerprint

- Python: `3.10.20`
- torch: `2.10.0+cu128`
- transformers: `5.3.0`
- vLLM: `0.17.1`
- numpy: `2.2.6`
- pandas: `2.3.3`

Validated on this host:

- OS/arch: `linux-64`
- GPU: `NVIDIA GeForce RTX 3090` x 8
- Driver: `580.82.09`
- CUDA: `13.0` (`nvidia-smi` 기준)

## Files

- Main conda env file: [environment_qwen_moe_host.yml](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/environment_qwen_moe_host.yml)
- Raw conda lock: [conda_explicit_qwen_moe_host.txt](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/conda_explicit_qwen_moe_host.txt)
- Pinned pip list: [requirements_qwen_moe_host.txt](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/requirements_qwen_moe_host.txt)
- Setup script: [setup_qwen_moe_host.sh](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/setup_qwen_moe_host.sh)
- Export refresh script: [export_qwen_moe_host_env.sh](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/export_qwen_moe_host_env.sh)
- Inference execution checklist: `QWEN_INFERENCE_EXECUTION_CHECKLIST.md`

## Recommended install

Conda env file install is the default.
`environment_qwen_moe_host.yml` already contains pinned conda builds and the pip section, so other people can recreate the environment with `conda env create`.

```bash
cd /path/to/qwen_tv
./setup_qwen_moe_host.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
```

Notes:

- Default `./setup_qwen_moe_host.sh` now means `INSTALL_MODE=conda_env`.
- The env file is intended for `linux-64`.
- This is the recommended way when you want other people to build the same environment with conda.
- Hugging Face model weights are not bundled here. First download still needs internet access.

## Raw lock install

Use this only when you want to restore from the lower-level raw lock files instead of the main conda env file.

```bash
cd /path/to/qwen_tv
INSTALL_MODE=lock ./setup_qwen_moe_host.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
```

## Manual install

Conda env file:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -n qwen_moe_host -f environment_qwen_moe_host.yml
conda activate qwen_moe_host
```

Raw lock:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n qwen_moe_host --file conda_explicit_qwen_moe_host.txt -y
conda activate qwen_moe_host
python -m pip install -r requirements_qwen_moe_host.txt
```

## Refreshing the lock files

If `qwen_moe_host` changes, regenerate the distributable files with:

```bash
cd /path/to/qwen_tv
./export_qwen_moe_host_env.sh
```

## Quick verification

```bash
conda activate qwen_moe_host
python -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

## Next Step

After the environment is created, check `QWEN_INFERENCE_EXECUTION_CHECKLIST.md` for the runtime requirements of:

- `inference_qwen35_9b_base_video_classification.py`
- `inference_qwen35_openai_vllm_classification.py`
