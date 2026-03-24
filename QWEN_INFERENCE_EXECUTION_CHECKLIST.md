# Qwen Inference Execution Checklist

This checklist is for other people who want to run the inference scripts in this repo after recreating the `qwen_moe_host` conda environment.

## Scope

- `inference_qwen35_9b_base_video_classification.py`
- `inference_qwen35_openai_vllm_classification.py`

## Shared Checklist

- [ ] Linux `x86_64` machine with NVIDIA GPU is available.
- [ ] `conda` is installed.
- [ ] This repo is checked out locally.
- [ ] The dataset exists under `qwen_tv/dataset` or the user will pass explicit paths.
- [ ] Internet access is available for the first Hugging Face model download, or the required models are already cached locally.
- [ ] The user has permission to write under `qwen_tv/result` and `qwen_tv/logs`.

## Conda Environment Checklist

- [ ] Create the conda environment from `environment_qwen_moe_host.yml`.

```bash
cd /path/to/qwen_tv
./setup_qwen_moe_host.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
```

- [ ] Verify the core packages import correctly.

```bash
python -c "import torch, transformers, vllm, openai, decord, cv2, pandas, numpy, PIL; print('imports_ok')"
```

- [ ] Verify the key runtime versions if needed.

```bash
python -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

## Dataset Path List Checklist

`utils.py` contains helper functions to generate `train_video_path.txt` and `test_video_path.txt`.

- [ ] Decide the root directory for the train videos.
- [ ] Decide the root directory for the test videos.
- [ ] Decide where the generated `.txt` files should be written.
- [ ] Generate `train_video_path.txt` if training or train-set evaluation needs it.
- [ ] Generate `test_video_path.txt` if batch inference needs it.

Recommended usage with explicit paths:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
python -c "from utils import make_train_video_path; make_train_video_path(input_file='dataset/sim_dataset/videos', output_file='dataset')"
python -c "from utils import make_test_video_path; make_test_video_path(input_file='dataset/videos', output_file='dataset')"
```

This writes:

- `dataset/train_video_path.txt`
- `dataset/test_video_path.txt`

If the local dataset layout already matches the hard-coded defaults inside `utils.py`, the user can also run:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
python utils.py
```

Notes:

- `make_train_video_path()` in `utils.py` scans `*.mp4` recursively and writes `train_video_path.txt`.
- `make_test_video_path()` in `utils.py` scans `*.mp4` recursively and writes `test_video_path.txt`.
- Most inference scripts use `dataset/test_video_path.txt` as the default batch input list.

## 9B Script Checklist

Target script:

- `inference_qwen35_9b_base_video_classification.py`

What this script expects:

- [ ] The conda environment is activated.
- [ ] A usable GPU is visible to PyTorch.
- [ ] The model `Qwen/Qwen3.5-9B` can be downloaded from Hugging Face, or is already cached locally.
- [ ] At least one of these inputs is available:
- [ ] `--video_path`
- [ ] `--video_list`
- [ ] `--base_submission_csv`
- [ ] If running without internet, the user passes `--local_files_only true` and the model is already cached.

Quick smoke test:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
python inference_qwen35_9b_base_video_classification.py \
  --video_path dataset/videos/-2UPLUV7JLg_00.mp4 \
  --experiment_name qwen35_9b_base_video_check
```

If the user prefers a batch run:

```bash
python inference_qwen35_9b_base_video_classification.py \
  --video_list dataset/test_video_path.txt \
  --experiment_name qwen35_9b_base_video_batch
```

## 27B vLLM Script Checklist

Target script:

- `inference_qwen35_openai_vllm_classification.py`

What this script expects:

- [ ] The conda environment is activated.
- [ ] A reachable OpenAI-compatible vLLM server exists at `OPENAI_BASE_URL`.
- [ ] A reachable HTTP server exists for the dataset root at `BASE_VIDEO_URL`, or the user otherwise provides a resolvable video URL path setup.
- [ ] The served model is compatible with the request, typically `Qwen/Qwen3.5-27B-FP8`.
- [ ] At least one of these inputs is available:
- [ ] `--video_path`
- [ ] `--video_list`
- [ ] `--base_submission_csv`

Recommended path:

- [ ] Use `run_openai_infer.sh` instead of calling the Python file directly.
- [ ] Prepare `.env` if the default ports or paths are not correct.

Minimal `.env` example:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
MODEL_ID=Qwen/Qwen3.5-27B-FP8
BASE_VIDEO_URL=http://localhost:8002
DATASET_ROOT=/path/to/qwen_tv/dataset
LOG_DIR=/path/to/qwen_tv/logs
VLLM_TP_SIZE=2
KEEP_SERVERS=0
```

Quick smoke test with the helper script:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_moe_host
CUDA_VISIBLE_DEVICES=6,7 ./run_openai_infer.sh \
  --video_path dataset/videos/-2UPLUV7JLg_00.mp4 \
  --experiment_name qwen35_27b_vllm_check
```

Direct Python invocation is also possible, but only if the vLLM endpoint and video server are already running:

```bash
python inference_qwen35_openai_vllm_classification.py \
  --video_path dataset/videos/-2UPLUV7JLg_00.mp4 \
  --openai_base_url http://localhost:8000/v1 \
  --base_video_url http://localhost:8002 \
  --experiment_name qwen35_27b_vllm_direct
```

## What The YAML Does And Does Not Guarantee

What `environment_qwen_moe_host.yml` does cover:

- [ ] Python package versions
- [ ] `conda` packages
- [ ] `pip` packages listed inside the environment file

What `environment_qwen_moe_host.yml` does not cover by itself:

- [ ] NVIDIA driver on the host machine
- [ ] CUDA driver compatibility on the host machine
- [ ] Hugging Face model weights cache
- [ ] The dataset files
- [ ] Running vLLM server processes
- [ ] Running HTTP file server processes
- [ ] GPU memory sufficiency for a chosen model

## Hand-Off Summary

If someone else wants to reproduce and run the scripts, the recommended hand-off set is:

- [ ] `environment_qwen_moe_host.yml`
- [ ] `setup_qwen_moe_host.sh`
- [ ] This checklist file
- [ ] `utils.py` if they need to regenerate `train_video_path.txt` or `test_video_path.txt`
- [ ] `run_openai_infer.sh` for the 27B/vLLM path
- [ ] The repo code itself
- [ ] Access to the dataset
