# inference.py Setup With qwen_test

`inference.py` is expected to run in the `qwen_test` conda environment.

## Environment

Create the environment:

```bash
cd /path/to/qwen_tv
./setup_qwen_test.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_test
```

Manual alternative:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -n qwen_test -f environment_qwen_test.yml
conda activate qwen_test
```

## Required Files And Runtime Assumptions

- `inference.py`
- `utils.py`
- `dataset/test_video_path.txt`
- local dataset video files
- at least one visible CUDA GPU

Notes:

- `inference.py` reads `dataset/test_video_path.txt` as the batch input file.
- `inference.py` uses one worker per visible GPU.
- If `HF_TOKEN` is set in `.env`, the script logs in to Hugging Face before loading the model.
- If `HF_TOKEN` is not set, the script continues without login. This is fine when the model is public or already cached.

## Optional .env

Create `.env` only if you need a Hugging Face token:

```bash
HF_TOKEN=your_huggingface_token
```

Do not commit real tokens to git.

## Build test_video_path.txt

Recommended:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_test
python -c "from utils import make_test_video_path; make_test_video_path(input_file='dataset/videos', output_file='dataset')"
```

This creates:

```text
dataset/test_video_path.txt
```

## Run inference.py

Example with two visible GPUs:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_test
CUDA_VISIBLE_DEVICES=0,1 python inference.py
```

Example with one visible GPU:

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_test
CUDA_VISIBLE_DEVICES=0 python inference.py
```

## Outputs

Running `inference.py` writes:

- parsed JSON files under `result/parsed_json`
- a timestamped submission directory under `result/`

The timestamped directory includes:

- `submission.csv`
- `description.txt`

## Quick Validation

```bash
cd /path/to/qwen_tv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_test
python -c "import torch, transformers, pandas, huggingface_hub; print('imports_ok')"
```
