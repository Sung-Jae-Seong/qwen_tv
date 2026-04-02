#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT_DIR/inference_qwen35_9b_base_video_classification_two_stage.py"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-9B}"
STAGE1_MODEL_NAME_OR_PATH="${STAGE1_MODEL_NAME_OR_PATH:-}"
STAGE2_MODEL_NAME_OR_PATH="${STAGE2_MODEL_NAME_OR_PATH:-}"
VIDEO_PATH="${VIDEO_PATH:-}"
VIDEO_LIST="${VIDEO_LIST:-$ROOT_DIR/dataset/test_video_path.txt}"
BASE_SUBMISSION_CSV="${BASE_SUBMISSION_CSV:-}"
STAGE1_PREDICTIONS_CSV="${STAGE1_PREDICTIONS_CSV:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
STAGE1_MAX_NEW_TOKENS="${STAGE1_MAX_NEW_TOKENS:-192}"
STAGE2_MAX_NEW_TOKENS="${STAGE2_MAX_NEW_TOKENS:-256}"
VIDEO_FPS="${VIDEO_FPS:-}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-5.0}"
USE_SUBMISSION_TIME_CLIP="${USE_SUBMISSION_TIME_CLIP:-true}"
CLIP_SECONDS_BEFORE="${CLIP_SECONDS_BEFORE:-3.0}"
CLIP_SECONDS_AFTER="${CLIP_SECONDS_AFTER:-3.0}"
FALLBACK_TO_EXISTING_TYPE="${FALLBACK_TO_EXISTING_TYPE:-true}"
LIMIT="${LIMIT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_base_video_classification_two_stage}"

if [[ -z "$VIDEO_PATH" && -z "$BASE_SUBMISSION_CSV" && ! -f "$VIDEO_LIST" ]]; then
    echo "No input source found." >&2
    echo "Set one of VIDEO_PATH, BASE_SUBMISSION_CSV, or create $VIDEO_LIST" >&2
    exit 1
fi

cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_PATH"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --local_files_only "$LOCAL_FILES_ONLY"
    --torch_dtype "$TORCH_DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --stage1_max_new_tokens "$STAGE1_MAX_NEW_TOKENS"
    --stage2_max_new_tokens "$STAGE2_MAX_NEW_TOKENS"
    --video_max_frames "$VIDEO_MAX_FRAMES"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_retries "$MAX_RETRIES"
    --retry_backoff_sec "$RETRY_BACKOFF_SEC"
    --use_submission_time_clip "$USE_SUBMISSION_TIME_CLIP"
    --clip_seconds_before "$CLIP_SECONDS_BEFORE"
    --clip_seconds_after "$CLIP_SECONDS_AFTER"
    --fallback_to_existing_type "$FALLBACK_TO_EXISTING_TYPE"
    --experiment_name "$EXPERIMENT_NAME"
)

if [[ -n "$STAGE1_MODEL_NAME_OR_PATH" ]]; then
    cmd+=(--stage1_model_name_or_path "$STAGE1_MODEL_NAME_OR_PATH")
fi
if [[ -n "$STAGE2_MODEL_NAME_OR_PATH" ]]; then
    cmd+=(--stage2_model_name_or_path "$STAGE2_MODEL_NAME_OR_PATH")
fi
if [[ -n "$VIDEO_PATH" ]]; then
    cmd+=(--video_path "$VIDEO_PATH")
fi
if [[ -n "$BASE_SUBMISSION_CSV" ]]; then
    cmd+=(--base_submission_csv "$BASE_SUBMISSION_CSV")
else
    cmd+=(--video_list "$VIDEO_LIST")
fi
if [[ -n "$STAGE1_PREDICTIONS_CSV" ]]; then
    cmd+=(--stage1_predictions_csv "$STAGE1_PREDICTIONS_CSV")
fi
if [[ -n "$VIDEO_FPS" ]]; then
    cmd+=(--video_fps "$VIDEO_FPS")
fi
if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
fi

cmd+=("$@")

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n\n'
"${cmd[@]}"
