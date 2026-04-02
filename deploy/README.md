# Qwen3.5-9B Two-Stage Inference Deploy Bundle

This folder is a portable bundle for running [inference_qwen35_9b_base_video_classification_two_stage.py](/home/milab/cvpr2026cctv/team01/minseok/qwen_tv/deploy/inference_qwen35_9b_base_video_classification_two_stage.py) on another machine.

It was prepared from the `qwen_moe_host` Conda environment that was used locally on April 2, 2026.

## What Is Included

- `inference_qwen35_9b_base_video_classification_two_stage.py`
  - Main two-stage inference entrypoint.
- `inference_qwen35_9b_base_video_classification.py`
  - Helper module imported by the main script.
- `environment_qwen_moe_host.yml`
  - Recommended Conda env file for reproduction.
- `conda_explicit_qwen_moe_host.txt`
  - Exact Conda package lock.
- `requirements_qwen_moe_host.txt`
  - Exact pip package lock for the same host env.
- `requirements_inference_core.txt`
  - Smaller curated dependency set for inference-only installs.
- `setup_qwen_moe_host.sh`
  - Env creation helper.
- `run_two_stage_inference.sh`
  - Convenience launcher.
- `check_inference_env.py`
  - Quick import and CUDA sanity check.
- `dataset/`
  - Place `test_video_path.txt` and videos here if you want the default paths to work.
- `examples/`
  - Example input files.
- `result/`
  - Default output root created by the scripts.

## Recommended Directory Layout

```text
deploy/
├── inference_qwen35_9b_base_video_classification_two_stage.py
├── inference_qwen35_9b_base_video_classification.py
├── environment_qwen_moe_host.yml
├── conda_explicit_qwen_moe_host.txt
├── requirements_qwen_moe_host.txt
├── requirements_inference_core.txt
├── setup_qwen_moe_host.sh
├── run_two_stage_inference.sh
├── check_inference_env.py
├── dataset/
│   ├── README.md
│   ├── test_video_path.txt
│   └── videos/
└── result/
```

## Environment Setup

### Option 1: Recommended

Use the same Conda env name that was used locally:

```bash
cd qwen_tv/deploy
bash setup_qwen_moe_host.sh
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate qwen_moe_host
python check_inference_env.py
```

This uses `environment_qwen_moe_host.yml` by default.

### Option 2: Exact Lock Recreation

```bash
cd qwen_tv/deploy
INSTALL_MODE=lock bash setup_qwen_moe_host.sh
```

This uses:

- `conda_explicit_qwen_moe_host.txt`
- `requirements_qwen_moe_host.txt`

### Option 3: Smaller Inference-Only Install

This is lighter, but less reproducible than the exact host env:

```bash
cd qwen_tv/deploy
INSTALL_MODE=core bash setup_qwen_moe_host.sh
```

## Model Setup

By default, the launcher uses the Hugging Face model id:

```text
Qwen/Qwen3.5-9B
```

If you want to run from a local model snapshot instead:

```bash
export MODEL_NAME_OR_PATH=/path/to/local/Qwen3.5-9B
export LOCAL_FILES_ONLY=true
```

If the model is not already cached locally, make sure the machine can download from Hugging Face, or pre-download the snapshot before setting `LOCAL_FILES_ONLY=true`.

## Input Modes

The two-stage script supports three input modes.

### 1. Single video

```bash
python inference_qwen35_9b_base_video_classification_two_stage.py \
  --video_path /abs/path/to/video.mp4 \
  --model_name_or_path Qwen/Qwen3.5-9B
```

### 2. Video list file

Default path:

```text
deploy/dataset/test_video_path.txt
```

Each line should be one video path. Relative paths such as `videos/example.mp4` are resolved against `deploy/dataset/`.

```bash
cp dataset/test_video_path.txt.example dataset/test_video_path.txt
python inference_qwen35_9b_base_video_classification_two_stage.py \
  --video_list dataset/test_video_path.txt \
  --model_name_or_path Qwen/Qwen3.5-9B
```

### 3. Base submission CSV

This is the most useful mode if you already have `accident_time` and want Stage 2 to classify only a clipped window around that time.

Required columns:

- `path`
- `type`

Recommended additional column:

- `accident_time`

Example:

```bash
python inference_qwen35_9b_base_video_classification_two_stage.py \
  --base_submission_csv examples/base_submission.example.csv \
  --model_name_or_path Qwen/Qwen3.5-9B
```

## Quick Start With The Launcher

The shell wrapper exposes common settings as environment variables and forwards any extra CLI args to the Python script.

```bash
cd qwen_tv/deploy
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate qwen_moe_host

export MODEL_NAME_OR_PATH=Qwen/Qwen3.5-9B
export VIDEO_LIST="$PWD/dataset/test_video_path.txt"

bash run_two_stage_inference.sh
```

Using a base submission CSV:

```bash
export BASE_SUBMISSION_CSV="$PWD/my_submission.csv"
bash run_two_stage_inference.sh
```

Reusing existing Stage 1 predictions:

```bash
export STAGE1_PREDICTIONS_CSV="$PWD/result/previous_run/stage1_group_predictions.csv"
bash run_two_stage_inference.sh
```

## Outputs

Every run creates:

```text
result/<experiment_name>_<timestamp>/
```

Common files inside:

- `type_predictions.csv`
- `stage1_group_predictions.csv`
- `submission_replace_only_type.csv`
- `submission_replace_only_type_stage1.csv`
- `description.txt`
- `raw_responses/`
- `parsed_predictions/`

## GPU Notes

- This script loads Qwen3.5-9B with `device_map="auto"`.
- A CUDA-capable PyTorch install is expected for practical runs.
- `check_inference_env.py` reports CUDA visibility and import status before you start a long inference run.

## Common Problems

### `ModuleNotFoundError: qwen_vl_utils`

The environment is incomplete. Recreate the env with:

```bash
bash setup_qwen_moe_host.sh
```

### `Path not found`

Check whether your video paths are:

- absolute paths, or
- relative to `deploy/dataset/`

### Stage 2 is not clipping around accident time

That only happens when:

- you use `--base_submission_csv`
- and the CSV has an `accident_time` column
- and `--use_submission_time_clip true`

## Recommended Files To Share

If you want to hand this to someone else, share the whole `deploy/` folder plus either:

- a local model snapshot path, or
- a note that the model should be downloaded from Hugging Face.

## Run command example
```bash
CUDA_VISIBLE_DEVICES=6,7 nohup python3 inference_qwen35_9b_base_video_classification_two_stage.py \
  --local_files_only true \
  --base_submission_csv /home/milab/cvpr2026cctv/team01/minseok/qwen_tv/submission_rian_standard.csv \
  --stage1_predictions_csv /home/milab/cvpr2026cctv/team01/minseok/qwen_tv/result/qwen35_9b_two_stage_stage1full_stage2clip_fps8_gpu67_v5_20260329110839/submission_replace_only_type_stage1.csv \
  --use_submission_time_clip true \
  --clip_seconds_before 3.0 \
  --clip_seconds_after 3.0 \
  --video_fps 8 \
  --video_max_frames 1024 \
  --stage2_max_new_tokens 96 \
  --experiment_name qwen35_9b_stage2_only_rerun_gpu67 \
  > /home/milab/cvpr2026cctv/team01/minseok/qwen_tv/infer_qwen35_9b_stage2_v5_only_rerun_gpu67.log 2>&1 &
```
