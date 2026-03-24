#!/usr/bin/env python3
"""
Qwen3.5-27B 계열을 vLLM OpenAI-compatible endpoint로 호출해서
비디오의 crash type만 분류한다.

기본 사용 예시:

python3 inference_qwen35_openai_vllm_classification.py \
  --base_submission_csv submission_qwen3.5_27b_vllm.csv \
  --base_video_url http://localhost:8002 \
  --openai_base_url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPO_ROOT / "result"
DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B-FP8"
DEFAULT_TEST_PATH_FILE = DATASET_ROOT / "test_video_path.txt"
DEFAULT_EXPERIMENT_NAME = "qwen35_openai_vllm_classification"
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
ACCIDENT_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone", "single"]
MIN_REASONING_MAX_TOKENS = 192
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SEC = 5.0

SYSTEM_PROMPT = """You are an AI vision expert classifying CCTV traffic crashes using ANSI D16.1 and FMCSA-2022 crash classification standards.

Your label must be based only on the first harmful event, meaning the exact first physical contact. Ignore approach behavior, near-misses, braking, swerving, and later post-impact motion.

Vehicle status:
- In-Transport: a vehicle that is moving, or stopped within the portion of the roadway ordinarily used by similar vehicles. This includes a disabled vehicle abandoned in an active travel lane.
- Not In-Transport: a vehicle that is legally parked off the roadway, or completely stopped on the shoulder, median, or roadside.

Crash labels:
- head-on: front-to-front contact between two In-Transport vehicles traveling in opposite directions
- rear-end: front of the trailing In-Transport vehicle strikes the rear of the leading In-Transport vehicle traveling in the same direction
- sideswipe: side-to-side glancing contact between two In-Transport vehicles traveling in the same or opposite directions
- t-bone: front of one In-Transport vehicle strikes the side of another In-Transport vehicle at or near a right angle
- single: exactly one In-Transport vehicle is involved in the first harmful event; if an In-Transport vehicle hits a Not In-Transport vehicle, the label is single

Rear-end vs sideswipe:
- If the first harmful event begins with the front of one vehicle hitting the rear or rear corner of another vehicle moving in the same direction, choose rear-end.
- If the first harmful event begins with side-to-side scraping, brushing, or sliding contact along the vehicle sides, choose sideswipe.
- Do not choose sideswipe when the initial impact is front-to-rear, even if the vehicles slide along each other afterward.

Decision rules:
1. Review the full video before deciding, but classify only the first harmful event.
2. First determine how many In-Transport vehicles are involved in that first physical contact.
3. If two In-Transport vehicles make first contact with each other, the answer must not be single.
4. Choose among head-on, rear-end, sideswipe, and t-bone strictly from the initial contact geometry.
5. Choose single only after ruling out those multi-vehicle labels.

Keep reasoning short and classification-focused. Do not narrate the whole video. Do not speculate about unseen impact.

Return only one JSON object matching this structure:
{
  "perception": {
    "global": "...",
    "local": "..."
  },
  "cognition": {
    "shallow": "...",
    "deep": "..."
  },
  "answer": {
    "type": "head-on | rear-end | sideswipe | t-bone | single",
    "why": "..."
  }
}

Field guidance:
- perception.global: one short sentence about the road scene before the crash
- perception.local: one short sentence naming the vehicles and where the first harmful event occurs
- cognition.shallow: one short sentence describing the visible first contact
- cognition.deep: one short sentence applying In-Transport status and impact geometry
- answer.why: one short sentence justifying the label

No markdown. No extra text."""

USER_PROMPT = (
    "Analyze the full video from first frame to last frame. Find the first harmful event, decide the crash type from "
    "that first contact only, and return exactly one JSON object. Keep every text field short and focused on the "
    "classification decision."
)

DIRECT_TYPE_MAP = {
    "headon": "head-on",
    "rearend": "rear-end",
    "sideswipe": "sideswipe",
    "single": "single",
    "singlevehicle": "single",
    "singlevehiclecrash": "single",
    "tbone": "t-bone",
    "tboneangle": "t-bone",
    "angle": "t-bone",
}

TYPE_PATTERNS = [
    ("head-on", re.compile(r"\bhead[\s-]?on\b", re.IGNORECASE)),
    ("rear-end", re.compile(r"\brear[\s-]?end\b", re.IGNORECASE)),
    ("sideswipe", re.compile(r"\bsideswipe\b", re.IGNORECASE)),
    ("t-bone", re.compile(r"\bt[\s-]?bone(?:\s*/\s*angle)?\b", re.IGNORECASE)),
    ("t-bone", re.compile(r"\bangle\b", re.IGNORECASE)),
]
EXPLICIT_TYPE_PATTERN = re.compile(
    r"(?:crash\s*type|classification|classified\s+as|classified\s+the\s+crash\s+as|answer|output)\s*"
    r"(?:is|:)?\s*"
    r"(head[\s-]?on|rear[\s-]?end|sideswipe|single(?:[\s-]?vehicle(?:[\s-]?crash)?)?|t[\s-]?bone(?:\s*/\s*angle)?|angle)\b",
    re.IGNORECASE,
)
TAIL_TYPE_PATTERN = re.compile(
    r"(head[\s-]?on|rear[\s-]?end|sideswipe|single|t[\s-]?bone(?:\s*/\s*angle)?|angle)\s*[\.\!\)\]]*\s*$",
    re.IGNORECASE,
)


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use one of true/false, yes/no, 1/0."
    )


def _atomic_write_text(path: Path, text: str):
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def ensure_run_dir(experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RESULT_ROOT / f"{experiment_name}_{timestamp}"
    (run_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed_predictions").mkdir(parents=True, exist_ok=True)
    return run_dir


def iter_candidate_paths(path_value: str, extra_bases: Optional[Iterable[Path]] = None):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        yield path
        return

    seen = set()
    bases = [Path.cwd(), REPO_ROOT, DATASET_ROOT]
    if extra_bases:
        bases.extend(Path(base) for base in extra_bases)

    for base in bases:
        candidate = (base / path).resolve()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate


def iter_relocated_absolute_paths(path_value: str, extra_bases: Optional[Iterable[Path]] = None):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return

    seen = set()
    bases = [Path.cwd(), REPO_ROOT, DATASET_ROOT]
    if extra_bases:
        bases.extend(Path(base) for base in extra_bases)

    suffixes = []
    parts = path.parts
    repo_name = REPO_ROOT.name

    if repo_name in parts:
        repo_index = parts.index(repo_name)
        if repo_index + 1 < len(parts):
            suffixes.append(Path(*parts[repo_index + 1 :]))

    for anchor in ("dataset", "videos", "result", "output"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            suffixes.append(Path(*parts[anchor_index:]))

    for base in bases:
        for suffix in suffixes:
            candidate = (base / suffix).resolve()
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                yield candidate


def find_existing_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> Optional[Path]:
    for candidate in iter_candidate_paths(path_value, extra_bases=extra_bases):
        if candidate.exists():
            return candidate

    relocated = iter_relocated_absolute_paths(path_value, extra_bases=extra_bases)
    if relocated is not None:
        for candidate in relocated:
            if candidate.exists():
                return candidate
    return None


def resolve_existing_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> Path:
    resolved = find_existing_path(path_value, extra_bases=extra_bases)
    if resolved is not None:
        return resolved

    searched = list(iter_candidate_paths(path_value, extra_bases=extra_bases))
    relocated = iter_relocated_absolute_paths(path_value, extra_bases=extra_bases)
    if relocated is not None:
        searched.extend(list(relocated))
    searched_text = ", ".join(str(item) for item in searched)
    raise FileNotFoundError(f"Path not found: {path_value}. Searched: {searched_text}")


def resolve_video_absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return resolve_existing_path(path_value)
    return resolve_existing_path(str(DATASET_ROOT / path))


def extract_text_content(message_content):
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        texts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts).strip()

    return str(message_content)


def resolve_video_url(video_path: str, base_video_url: Optional[str] = None) -> str:
    if base_video_url:
        normalized_base = base_video_url.rstrip("/")
        path_obj = Path(video_path)
        if "videos" in path_obj.parts:
            relative_parts = path_obj.parts[path_obj.parts.index("videos") :]
            relative_path = "/".join(relative_parts)
        else:
            relative_path = path_obj.name
        return f"{normalized_base}/{relative_path}"

    return f"file://{os.path.abspath(video_path)}"


def to_submission_path(video_path: str) -> str:
    normalized = str(video_path)
    if "/videos/" in normalized:
        return "videos/" + normalized.split("/videos/")[-1]
    return normalized


def read_base_submission(base_submission_csv: Path) -> pd.DataFrame:
    submission_df = pd.read_csv(base_submission_csv)
    required = {"path", "type"}
    if not required.issubset(submission_df.columns):
        raise ValueError(
            f"Base submission CSV must contain at least {sorted(required)}: {base_submission_csv}"
        )
    return submission_df


def load_jobs(
    *,
    video_path: Optional[str],
    video_list: Optional[str],
    base_submission_csv: Optional[str],
    limit: Optional[int],
) -> Tuple[List[Dict[str, str]], Optional[pd.DataFrame], Optional[Path]]:
    if video_path:
        resolved = resolve_existing_path(video_path)
        jobs = [{"path": to_submission_path(str(resolved)), "video_path": str(resolved)}]
        return jobs, None, None

    if base_submission_csv:
        base_submission_path = resolve_existing_path(base_submission_csv)
        submission_df = read_base_submission(base_submission_path)
        if limit is not None:
            submission_df = submission_df.head(limit)
        jobs = []
        for row in submission_df.to_dict("records"):
            relative_path = str(row.get("path", "")).strip()
            resolved = resolve_video_absolute_path(relative_path)
            jobs.append({"path": relative_path, "video_path": str(resolved)})
        return jobs, submission_df, base_submission_path

    target_list = resolve_existing_path(video_list or str(DEFAULT_TEST_PATH_FILE))
    jobs = []
    with target_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            resolved = resolve_existing_path(stripped, extra_bases=[target_list.parent])
            jobs.append({"path": to_submission_path(str(resolved)), "video_path": str(resolved)})
    if limit is not None:
        jobs = jobs[:limit]
    return jobs, None, None


def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(candidate)
    except Exception:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if code_block:
        cleaned = code_block.group(1).strip()

    direct = try_parse_single_json(cleaned)
    if direct is not None:
        return direct

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        return try_parse_single_json(json_match.group(0))
    return None


def try_parse_type_from_json(text: str) -> Optional[str]:
    direct = extract_json_dict(text)
    if not isinstance(direct, dict):
        return None

    for key in ("type", "crash_type", "classification", "label"):
        normalized = normalize_accident_type(direct.get(key))
        if normalized is not None:
            return normalized

    answer = direct.get("answer")
    if isinstance(answer, dict):
        for key in ("type", "which", "crash_type", "classification", "label"):
            normalized = normalize_accident_type(answer.get(key))
            if normalized is not None:
                return normalized
    return None


def normalize_accident_type(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    search_text = re.sub(r"[*_`]+", "", text)
    stripped_prefix = re.sub(
        r"^\s*(?:crash\s*type|classification|answer)\s*[:\-]\s*",
        "",
        search_text,
        flags=re.IGNORECASE,
    ).strip()

    direct_key = re.sub(r"[^a-z]+", "", search_text.lower())
    if direct_key in DIRECT_TYPE_MAP:
        return DIRECT_TYPE_MAP[direct_key]

    stripped_key = re.sub(r"[^a-z]+", "", stripped_prefix.lower())
    if stripped_key in DIRECT_TYPE_MAP:
        return DIRECT_TYPE_MAP[stripped_key]

    json_type = try_parse_type_from_json(text)
    if json_type is not None:
        return json_type

    explicit_matches = list(EXPLICIT_TYPE_PATTERN.finditer(search_text))
    if explicit_matches:
        explicit_key = re.sub(r"[^a-z]+", "", explicit_matches[-1].group(1).lower())
        return DIRECT_TYPE_MAP.get(explicit_key)

    tail_match = TAIL_TYPE_PATTERN.search(stripped_prefix)
    if tail_match:
        tail_key = re.sub(r"[^a-z]+", "", tail_match.group(1).lower())
        return DIRECT_TYPE_MAP.get(tail_key)

    matches: List[Tuple[int, str]] = []
    for accident_type, pattern in TYPE_PATTERNS:
        for match in pattern.finditer(search_text):
            matches.append((match.start(), accident_type))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def extract_structured_output_details(text: str) -> Dict[str, Any]:
    direct = extract_json_dict(text)
    if not isinstance(direct, dict):
        return {
            "structured_output": None,
            "predicted_type": None,
            "answer_why": "",
        }

    answer = direct.get("answer")
    answer_why = ""
    if isinstance(answer, dict):
        answer_why = str(answer.get("why", "") or "").strip()

    return {
        "structured_output": direct,
        "predicted_type": try_parse_type_from_json(text),
        "answer_why": answer_why,
    }


def save_raw_and_parsed_response(run_dir: Path, path_value: str, raw_text: str, parsed: Dict[str, Any]):
    stem = Path(path_value).stem or "unknown"
    _atomic_write_text(run_dir / "raw_responses" / f"{stem}.txt", raw_text)
    _atomic_write_json(run_dir / "parsed_predictions" / f"{stem}.json", parsed)


def make_type_predictions_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        results,
        columns=[
            "path",
            "video_path",
            "resolved_video_url",
            "predicted_type",
            "status",
            "answer_why",
            "structured_output",
            "raw_output",
        ],
    )


def summarize_status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in results:
        status = str(item.get("status", "") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "traffic_crash_reasoned_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "perception": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "global": {"type": "string"},
                            "local": {"type": "string"},
                        },
                        "required": ["global", "local"],
                    },
                    "cognition": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shallow": {"type": "string"},
                            "deep": {"type": "string"},
                        },
                        "required": ["shallow", "deep"],
                    },
                    "answer": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ACCIDENT_TYPES,
                            },
                            "why": {"type": "string"},
                        },
                        "required": ["type", "why"],
                    },
                },
                "required": ["perception", "cognition", "answer"],
            },
        },
    }


def replace_type_in_submission(
    base_submission_df: pd.DataFrame,
    type_by_path: Dict[str, str],
    *,
    fallback_to_existing_type: bool,
) -> pd.DataFrame:
    output_df = base_submission_df.copy()
    replaced_types = []
    for row in output_df.to_dict("records"):
        path_value = str(row.get("path", "")).strip()
        predicted = type_by_path.get(path_value, "")
        if predicted:
            replaced_types.append(predicted)
            continue
        if fallback_to_existing_type:
            replaced_types.append(row.get("type", ""))
        else:
            replaced_types.append("")
    output_df["type"] = replaced_types
    return output_df


class Qwen35VllmClassificationInference:
    def __init__(self, model_id: str, openai_base_url: str, openai_api_key: str):
        self.model_id = model_id
        self.client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)

    def video_inference(
        self,
        *,
        video_path: str,
        base_video_url: Optional[str],
        fps: Optional[float],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        guided_choice: bool,
        max_retries: int,
        retry_backoff_sec: float,
    ) -> Tuple[str, str]:
        video_url = resolve_video_url(video_path, base_video_url=base_video_url)
        effective_max_tokens = max(max_tokens, MIN_REASONING_MAX_TOKENS)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ]

        extra_body = {"top_k": top_k}
        if fps is not None and fps > 0:
            extra_body["mm_processor_kwargs"] = {
                "fps": fps,
                "do_sample_frames": True,
            }
        if guided_choice:
            extra_body["guided_choice"] = ACCIDENT_TYPES

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    response_format=build_response_format(),
                    extra_body=extra_body,
                )
                return extract_text_content(response.choices[0].message.content), video_url
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                sleep_seconds = max(0.0, retry_backoff_sec) * attempt
                print(
                    f"request failed for {video_path} on attempt {attempt}/{max_retries}: {repr(exc)}; "
                    f"retrying in {sleep_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_seconds)

        assert last_error is not None
        raise last_error


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM classification-only inference for Qwen video models")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--openai_base_url", type=str, default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--openai_api_key", type=str, default=DEFAULT_OPENAI_API_KEY)
    parser.add_argument("--video_path", type=str, default=None, help="single local video path")
    parser.add_argument("--video_list", type=str, default=str(DEFAULT_TEST_PATH_FILE))
    parser.add_argument(
        "--base_submission_csv",
        type=str,
        default=None,
        help="Optional submission CSV whose type column will be replaced using the new predictions.",
    )
    parser.add_argument(
        "--base_video_url",
        type=str,
        default=None,
        help="Base URL serving the dataset root. Example: http://localhost:8002",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="If set to a positive value, request video frame sampling at this FPS. Omit or use <= 0 to avoid forcing client-side FPS sampling.",
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--guided_choice", type=str2bool, default=False)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_backoff_sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--fallback_to_existing_type", type=str2bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main():
    args = parse_args()

    jobs, base_submission_df, resolved_base_submission_csv = load_jobs(
        video_path=args.video_path,
        video_list=args.video_list,
        base_submission_csv=args.base_submission_csv,
        limit=args.limit,
    )
    if not jobs:
        raise ValueError("No jobs found. Check --video_path, --video_list, or --base_submission_csv.")

    run_dir = ensure_run_dir(args.experiment_name)

    print("=" * 80, flush=True)
    print("Qwen3.5-27B vLLM classification-only inference", flush=True)
    print("=" * 80, flush=True)
    print(f"Model ID: {args.model_id}", flush=True)
    print(f"OpenAI base URL: {args.openai_base_url}", flush=True)
    print(f"Jobs: {len(jobs)}", flush=True)
    if resolved_base_submission_csv is not None:
        print(f"Base submission CSV: {resolved_base_submission_csv}", flush=True)
    print(f"Requested FPS sampling: {args.fps if args.fps is not None and args.fps > 0 else 'disabled'}", flush=True)
    print(f"Max retries: {args.max_retries}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    if args.guided_choice:
        print("Note: current vLLM server may ignore guided_choice; JSON schema remains the primary output control.", flush=True)

    inference = Qwen35VllmClassificationInference(
        model_id=args.model_id,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
    )

    results: List[Dict[str, Any]] = []
    type_by_path: Dict[str, str] = {}
    total_start_time = time.time()

    for index, job in enumerate(jobs, start=1):
        path_value = job["path"]
        video_path = job["video_path"]
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing {index}/{len(jobs)}: {Path(video_path).name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        status = "ok"
        raw_text = ""
        predicted_type = ""
        resolved_video_url = ""
        answer_why = ""
        structured_output = None

        try:
            raw_text, resolved_video_url = inference.video_inference(
                video_path=video_path,
                base_video_url=args.base_video_url,
                fps=args.fps,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                guided_choice=args.guided_choice,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
            details = extract_structured_output_details(raw_text)
            structured_output = details["structured_output"]
            answer_why = details["answer_why"]
            normalized = details["predicted_type"]
            if normalized is None:
                status = "parse_failed"
            else:
                predicted_type = normalized
        except Exception as exc:
            status = "error"
            raw_text = f"{type(exc).__name__}: {exc}"
            print(f"failed to process {video_path}: {repr(exc)}", flush=True)

        parsed = {
            "path": path_value,
            "video_path": video_path,
            "resolved_video_url": resolved_video_url,
            "predicted_type": predicted_type,
            "status": status,
            "answer_why": answer_why,
            "structured_output": structured_output,
            "raw_output": raw_text,
        }
        save_raw_and_parsed_response(run_dir, path_value, raw_text, parsed)
        results.append(parsed)
        type_by_path[path_value] = predicted_type

        print("raw:", flush=True)
        print(raw_text, flush=True)
        print(
            json.dumps(
                {
                    "path": path_value,
                    "predicted_type": predicted_type,
                    "status": status,
                    "answer_why": answer_why,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    total_elapsed = time.time() - total_start_time
    status_counts = summarize_status_counts(results)
    ok_count = status_counts.get("ok", 0)
    parse_failed_count = status_counts.get("parse_failed", 0)
    error_count = status_counts.get("error", 0)
    print(f"\nTotal elapsed time: {total_elapsed:.2f} seconds", flush=True)
    print(
        f"Completed {len(results)} jobs: ok={ok_count}, parse_failed={parse_failed_count}, error={error_count}",
        flush=True,
    )

    type_predictions_df = make_type_predictions_df(results)
    type_predictions_path = run_dir / "type_predictions.csv"
    type_predictions_df.to_csv(type_predictions_path, index=False, lineterminator="\n")

    description = (
        f"Model: {args.model_id}\n"
        f"OpenAI base URL: {args.openai_base_url}\n"
        f"Jobs: {len(results)}\n"
        f"Elapsed: {total_elapsed:.2f}s\n"
        f"fps: {args.fps if args.fps is not None and args.fps > 0 else 'disabled'}\n"
        f"max_tokens: {args.max_tokens}\n"
        f"effective_max_tokens: {max(args.max_tokens, MIN_REASONING_MAX_TOKENS)}\n"
        f"temperature: {args.temperature}\n"
        f"top_p: {args.top_p}\n"
        f"top_k: {args.top_k}\n"
        f"presence_penalty: {args.presence_penalty}\n"
        f"guided_choice: {args.guided_choice}\n"
        f"max_retries: {args.max_retries}\n"
        f"retry_backoff_sec: {args.retry_backoff_sec}\n"
        f"base_video_url: {args.base_video_url}\n"
        f"base_submission_csv: {resolved_base_submission_csv}\n"
        f"fallback_to_existing_type: {args.fallback_to_existing_type}\n"
        f"status_counts: {json.dumps(status_counts, ensure_ascii=False)}\n"
    )
    _atomic_write_text(run_dir / "description.txt", description)

    if base_submission_df is not None:
        replaced_submission_df = replace_type_in_submission(
            base_submission_df,
            type_by_path,
            fallback_to_existing_type=args.fallback_to_existing_type,
        )
        submission_out_path = run_dir / "submission_replace_only_type.csv"
        replaced_submission_df.to_csv(submission_out_path, index=False, lineterminator="\n")
        print(f"Saved replaced submission to: {submission_out_path}", flush=True)
        if error_count or parse_failed_count:
            print(
                "Warning: replaced submission may include fallback types for failed samples. "
                "Check type_predictions.csv before using it as final output.",
                flush=True,
            )

    print(f"Saved type predictions to: {type_predictions_path}", flush=True)
    print(f"Saved run artifacts to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
