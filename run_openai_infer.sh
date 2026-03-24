#!/usr/bin/env bash

# 이 스크립트는 .env를 먼저 읽고, OPENAI_BASE_URL에 붙는 vLLM 서버와 base_video_url용 HTTP 서버가 없으면 자동으로 띄운 뒤 inference_qwen35_openai.py를 실행합니다.
#
# ./run_openai_infer.sh \
#   --video_path /workspace/minseok/qwen_tv/dataset/videos/YXj8m_rAyaE_00.mp4 \
#   --model_id Qwen/Qwen3.5-27B-FP8 \
#   --base_video_url http://localhost:8002 \
#   --fps 2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OPENAI_BASE_URL_INCOMING="${OPENAI_BASE_URL-}"
OPENAI_API_KEY_INCOMING="${OPENAI_API_KEY-}"
MODEL_ID_INCOMING="${MODEL_ID-}"
BASE_VIDEO_URL_INCOMING="${BASE_VIDEO_URL-}"
DATASET_ROOT_INCOMING="${DATASET_ROOT-}"
LOG_DIR_INCOMING="${LOG_DIR-}"
VLLM_TP_SIZE_INCOMING="${VLLM_TP_SIZE-}"
KEEP_SERVERS_INCOMING="${KEEP_SERVERS-}"
VLLM_HEALTH_RETRIES_INCOMING="${VLLM_HEALTH_RETRIES-}"
VLLM_HEALTH_SLEEP_SECONDS_INCOMING="${VLLM_HEALTH_SLEEP_SECONDS-}"
HTTP_HEALTH_RETRIES_INCOMING="${HTTP_HEALTH_RETRIES-}"
HTTP_HEALTH_SLEEP_SECONDS_INCOMING="${HTTP_HEALTH_SLEEP_SECONDS-}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -n "${OPENAI_BASE_URL_INCOMING}" ]]; then export OPENAI_BASE_URL="${OPENAI_BASE_URL_INCOMING}"; fi
if [[ -n "${OPENAI_API_KEY_INCOMING}" ]]; then export OPENAI_API_KEY="${OPENAI_API_KEY_INCOMING}"; fi
if [[ -n "${MODEL_ID_INCOMING}" ]]; then export MODEL_ID="${MODEL_ID_INCOMING}"; fi
if [[ -n "${BASE_VIDEO_URL_INCOMING}" ]]; then export BASE_VIDEO_URL="${BASE_VIDEO_URL_INCOMING}"; fi
if [[ -n "${DATASET_ROOT_INCOMING}" ]]; then export DATASET_ROOT="${DATASET_ROOT_INCOMING}"; fi
if [[ -n "${LOG_DIR_INCOMING}" ]]; then export LOG_DIR="${LOG_DIR_INCOMING}"; fi
if [[ -n "${VLLM_TP_SIZE_INCOMING}" ]]; then export VLLM_TP_SIZE="${VLLM_TP_SIZE_INCOMING}"; fi
if [[ -n "${KEEP_SERVERS_INCOMING}" ]]; then export KEEP_SERVERS="${KEEP_SERVERS_INCOMING}"; fi
if [[ -n "${VLLM_HEALTH_RETRIES_INCOMING}" ]]; then export VLLM_HEALTH_RETRIES="${VLLM_HEALTH_RETRIES_INCOMING}"; fi
if [[ -n "${VLLM_HEALTH_SLEEP_SECONDS_INCOMING}" ]]; then export VLLM_HEALTH_SLEEP_SECONDS="${VLLM_HEALTH_SLEEP_SECONDS_INCOMING}"; fi
if [[ -n "${HTTP_HEALTH_RETRIES_INCOMING}" ]]; then export HTTP_HEALTH_RETRIES="${HTTP_HEALTH_RETRIES_INCOMING}"; fi
if [[ -n "${HTTP_HEALTH_SLEEP_SECONDS_INCOMING}" ]]; then export HTTP_HEALTH_SLEEP_SECONDS="${HTTP_HEALTH_SLEEP_SECONDS_INCOMING}"; fi

# Prefer shared libraries from the currently activated conda environment.
# This avoids picking an older system libstdc++ when vLLM imports sqlite/icu.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL_ID_DEFAULT="${MODEL_ID:-Qwen/Qwen3.5-27B-FP8}"
BASE_VIDEO_URL_DEFAULT="${BASE_VIDEO_URL:-http://localhost:8002}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/dataset}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
VLLM_HEALTH_RETRIES="${VLLM_HEALTH_RETRIES:-300}"
VLLM_HEALTH_SLEEP_SECONDS="${VLLM_HEALTH_SLEEP_SECONDS:-2}"
HTTP_HEALTH_RETRIES="${HTTP_HEALTH_RETRIES:-20}"
HTTP_HEALTH_SLEEP_SECONDS="${HTTP_HEALTH_SLEEP_SECONDS:-1}"

mkdir -p "$LOG_DIR"

VLLM_PID=""
HTTP_PID=""
STARTED_VLLM=0
STARTED_HTTP=0

has_arg() {
    local target="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

get_arg_value() {
    local target="$1"
    shift
    local previous=""
    local arg
    for arg in "$@"; do
        if [[ "$previous" == "$target" ]]; then
            printf "%s\n" "$arg"
            return 0
        fi
        previous="$arg"
    done
    return 1
}

extract_port() {
    local url="$1"
    local without_scheme="${url#*://}"
    local host_port="${without_scheme%%/*}"
    printf "%s\n" "${host_port##*:}"
}

extract_host() {
    local url="$1"
    local without_scheme="${url#*://}"
    local host_port="${without_scheme%%/*}"
    printf "%s\n" "${host_port%%:*}"
}

is_local_url() {
    local host
    host="$(extract_host "$1")"
    [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "0.0.0.0" ]]
}

wait_for_url() {
    local url="$1"
    local name="$2"
    local retries="${3:-120}"
    local sleep_seconds="${4:-2}"
    local i

    for ((i = 1; i <= retries; i++)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done

    echo "$name did not become ready: $url" >&2
    return 1
}

cleanup() {
    if [[ "$KEEP_SERVERS" == "1" ]]; then
        return
    fi

    if [[ "$STARTED_HTTP" == "1" && -n "$HTTP_PID" ]]; then
        kill "$HTTP_PID" >/dev/null 2>&1 || true
    fi

    if [[ "$STARTED_VLLM" == "1" && -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

REQUESTED_MODEL_ID="$(get_arg_value --model_id "$@" || true)"
MODEL_TO_SERVE="${REQUESTED_MODEL_ID:-$MODEL_ID_DEFAULT}"

REQUESTED_BASE_VIDEO_URL="$(get_arg_value --base_video_url "$@" || true)"
BASE_VIDEO_URL_TO_USE="${REQUESTED_BASE_VIDEO_URL:-$BASE_VIDEO_URL_DEFAULT}"

OPENAI_ROOT="${OPENAI_BASE_URL%/v1}"
VLLM_HEALTH_URL="${OPENAI_ROOT}/health"
VLLM_PORT="$(extract_port "$OPENAI_ROOT")"
HTTP_PORT="$(extract_port "$BASE_VIDEO_URL_TO_USE")"

echo "Project root: $ROOT_DIR"
echo "OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "Model: $MODEL_TO_SERVE"

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required but not installed." >&2
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm is required but not installed in the current environment." >&2
    exit 1
fi

if ! curl -fsS "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
    if ! is_local_url "$OPENAI_ROOT"; then
        echo "OPENAI_BASE_URL is not local and is not reachable: $OPENAI_BASE_URL" >&2
        exit 1
    fi

    echo "Starting vLLM server on port $VLLM_PORT"
    nohup vllm serve "$MODEL_TO_SERVE" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --enforce-eager \
        --disable-custom-all-reduce \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        >"$LOG_DIR/vllm_openai.log" 2>&1 &
    VLLM_PID="$!"
    STARTED_VLLM=1
    wait_for_url "$VLLM_HEALTH_URL" "vLLM server" "$VLLM_HEALTH_RETRIES" "$VLLM_HEALTH_SLEEP_SECONDS"
else
    echo "Using existing vLLM server at $OPENAI_BASE_URL"
fi

if ! curl -fsS "$BASE_VIDEO_URL_TO_USE/" >/dev/null 2>&1; then
    if ! is_local_url "$BASE_VIDEO_URL_TO_USE"; then
        echo "base_video_url is not local and is not reachable: $BASE_VIDEO_URL_TO_USE" >&2
        exit 1
    fi

    echo "Starting HTTP video server on port $HTTP_PORT"
    nohup python -m http.server "$HTTP_PORT" \
        --bind 0.0.0.0 \
        --directory "$DATASET_ROOT" \
        >"$LOG_DIR/video_http.log" 2>&1 &
    HTTP_PID="$!"
    STARTED_HTTP=1
    wait_for_url "$BASE_VIDEO_URL_TO_USE/" "HTTP video server" "$HTTP_HEALTH_RETRIES" "$HTTP_HEALTH_SLEEP_SECONDS"
else
    echo "Using existing video HTTP server at $BASE_VIDEO_URL_TO_USE"
fi

if [[ -n "$REQUESTED_BASE_VIDEO_URL" ]]; then
    echo "Using user-provided base_video_url: $BASE_VIDEO_URL_TO_USE"
else
    echo "Using auto-configured base_video_url: $BASE_VIDEO_URL_TO_USE"
fi

INFER_ARGS=("$@")

if ! has_arg --model_id "$@"; then
    INFER_ARGS+=("--model_id" "$MODEL_TO_SERVE")
fi

if ! has_arg --base_video_url "$@"; then
    INFER_ARGS+=("--base_video_url" "$BASE_VIDEO_URL_TO_USE")
fi

if ! has_arg --fps "$@"; then
    INFER_ARGS+=("--fps" "2")
fi

echo "Running inference_qwen35_openai.py"
python inference_qwen35_openai.py "${INFER_ARGS[@]}"
