#!/usr/bin/env python3
"""
Base Qwen/Qwen3.5-9B로 full video를 direct path로 넣어 사고 type만 분류한다.

이 스크립트는 코드에서 fps / max_frames를 명시하지 않는다.
즉, 우리 쪽에서 추가적인 video sampling 제한을 강제로 걸지 않는다.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


REPO_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPO_ROOT / "result"
DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_TEST_PATH_FILE = DATASET_ROOT / "test_video_path.txt"
DEFAULT_EXPERIMENT_NAME = "qwen35_9b_base_video_classification"
ACCIDENT_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone", "single"]
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SEC = 5.0
ASSISTANT_JSON_PREFIX = '{\n  "perception": {\n'

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

    searched_paths = list(iter_candidate_paths(path_value, extra_bases=extra_bases))
    relocated = iter_relocated_absolute_paths(path_value, extra_bases=extra_bases)
    if relocated is not None:
        searched_paths.extend(list(relocated))
    searched = ", ".join(str(path) for path in searched_paths)
    raise FileNotFoundError(f"Path not found: {path_value}. Searched: {searched}")


def resolve_video_absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return resolve_existing_path(path_value)
    return resolve_existing_path(str(DATASET_ROOT / path))


def to_submission_path(video_path: str) -> str:
    normalized = str(video_path)
    if "/videos/" in normalized:
        return "videos/" + normalized.split("/videos/")[-1]
    return normalized


def ensure_run_dir(experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RESULT_ROOT / f"{experiment_name}_{timestamp}"
    (run_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed_predictions").mkdir(parents=True, exist_ok=True)
    return run_dir


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


def maybe_restore_prefilled_json(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return text
    if stripped.startswith('"global"') or stripped.startswith('"local"'):
        return ASSISTANT_JSON_PREFIX + text
    return text


def normalize_accident_type(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    direct_key = re.sub(r"[^a-z]+", "", text.lower())
    if direct_key in DIRECT_TYPE_MAP:
        return DIRECT_TYPE_MAP[direct_key]

    structured = try_parse_type_from_json(text)
    if structured is not None:
        return structured

    matches: List[Tuple[int, str]] = []
    for accident_type, pattern in TYPE_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), accident_type))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def try_parse_type_from_json(text: str) -> Optional[str]:
    direct = extract_json_dict(text)
    if not isinstance(direct, dict):
        direct = extract_json_dict(maybe_restore_prefilled_json(text))
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


def extract_structured_output_details(text: str) -> Dict[str, Any]:
    direct = extract_json_dict(text)
    if not isinstance(direct, dict):
        direct = extract_json_dict(maybe_restore_prefilled_json(text))
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


def resolve_generation_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if mapped_device in (None, "cpu", "disk"):
                continue
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            return torch.device(str(mapped_device))

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_inputs_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def sanitize_processor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        if key in {"fps", "min_pixels", "max_pixels", "total_pixels", "max_frames"}:
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
        sanitized[key] = value
    return sanitized


def maybe_expand_video_grid_for_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if "video_grid_thw" not in inputs or "mm_token_type_ids" not in inputs:
        return inputs

    video_grid_thw = inputs["video_grid_thw"]
    mm_token_type_ids = inputs["mm_token_type_ids"]
    if not isinstance(video_grid_thw, torch.Tensor) or not isinstance(mm_token_type_ids, torch.Tensor):
        return inputs
    if video_grid_thw.numel() == 0:
        return inputs

    total_frame_count = int(video_grid_thw[:, 0].sum().item())
    mm_groups = 0
    for batch_index in range(mm_token_type_ids.shape[0]):
        previous = None
        for token_type in mm_token_type_ids[batch_index].tolist():
            if token_type == 2 and previous != 2:
                mm_groups += 1
            previous = token_type

    if mm_groups <= video_grid_thw.shape[0]:
        return inputs
    if mm_groups != total_frame_count:
        return inputs

    expanded_rows = []
    for t, h, w in video_grid_thw.tolist():
        expanded_rows.extend([[1, h, w] for _ in range(int(t))])

    inputs["video_grid_thw"] = torch.tensor(
        expanded_rows,
        dtype=video_grid_thw.dtype,
        device=video_grid_thw.device,
    )
    return inputs


def generate_and_decode(
    model,
    processor,
    prepared_inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    do_sample = temperature > 0
    generation_kwargs = dict(
        **prepared_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_token_id,
    )
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = model.generate(**generation_kwargs)

    prompt_length = prepared_inputs["input_ids"].shape[-1]
    generated_trimmed = generated_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def build_direct_video_messages(video_path: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": ASSISTANT_JSON_PREFIX,
        },
    ]


def save_raw_and_parsed_response(run_dir: Path, path_value: str, raw_text: str, parsed: Dict[str, Any]):
    stem = Path(path_value).stem or "unknown"
    with open(run_dir / "raw_responses" / f"{stem}.txt", "w", encoding="utf-8") as handle:
        handle.write(raw_text)
    with open(run_dir / "parsed_predictions" / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)


def make_type_predictions_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        results,
        columns=[
            "path",
            "video_path",
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


class Qwen35BaseVideoClassificationRunner:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        local_files_only: bool,
        torch_dtype: str,
        attn_implementation: str,
    ):
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError("--torch_dtype must be one of: auto, float16, bfloat16, float32")

        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }

        if dtype_map[torch_dtype] == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype_map[torch_dtype]

        from_pretrained_signature = inspect.signature(
            AutoModelForImageTextToText.from_pretrained
        ).parameters
        if attn_implementation.lower() != "auto" and "attn_implementation" in from_pretrained_signature:
            model_kwargs["attn_implementation"] = attn_implementation

        print(f"Loading processor from: {model_name_or_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        print(f"Loading model from: {model_name_or_path}", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        if getattr(self.processor, "tokenizer", None) is not None and self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.input_device = resolve_generation_device(self.model)
        print(f"Input device: {self.input_device}", flush=True)

    def infer(self, *, video_path: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        messages = build_direct_video_messages(video_path)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        image_inputs, video_inputs_with_metadata, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_kwargs = sanitize_processor_kwargs(video_kwargs)
        video_inputs = None
        video_metadata = None
        if video_inputs_with_metadata is not None:
            video_inputs = [item[0] for item in video_inputs_with_metadata]
            video_metadata = [item[1] for item in video_inputs_with_metadata]
        prepared = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            truncation=False,
            **video_kwargs,
        )
        prepared = maybe_expand_video_grid_for_generation(prepared)
        prepared = move_inputs_to_device(prepared, self.input_device)
        return generate_and_decode(
            model=self.model,
            processor=self.processor,
            prepared_inputs=prepared,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Base Qwen3.5-9B full-video classification inference")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_list", type=str, default=str(DEFAULT_TEST_PATH_FILE))
    parser.add_argument("--base_submission_csv", type=str, default=None)
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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
    type_predictions_path = run_dir / "type_predictions.csv"
    description_path = run_dir / "description.txt"

    print("=" * 80, flush=True)
    print("Qwen3.5-9B base full-video classification inference", flush=True)
    print("=" * 80, flush=True)
    print(f"Model: {args.model_name_or_path}", flush=True)
    print("Video sampling override: disabled in this script", flush=True)
    print(f"Jobs: {len(jobs)}", flush=True)
    if resolved_base_submission_csv is not None:
        print(f"Base submission CSV: {resolved_base_submission_csv}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)

    runner = Qwen35BaseVideoClassificationRunner(
        model_name_or_path=args.model_name_or_path,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    total_start_time = time.time()
    results: List[Dict[str, Any]] = []
    type_by_path: Dict[str, str] = {}

    for index, job in enumerate(jobs, start=1):
        path_value = job["path"]
        video_path = job["video_path"]
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing {index}/{len(jobs)}: {Path(video_path).name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        status = "ok"
        raw_text = ""
        predicted_type = ""
        answer_why = ""
        structured_output = None
        last_error: Optional[Exception] = None

        for attempt in range(1, args.max_retries + 1):
            try:
                raw_text = runner.infer(
                    video_path=video_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                raw_text = maybe_restore_prefilled_json(raw_text)
                details = extract_structured_output_details(raw_text)
                structured_output = details["structured_output"]
                answer_why = details["answer_why"]
                normalized = details["predicted_type"]
                if normalized is None:
                    status = "parse_failed"
                else:
                    predicted_type = normalized
                break
            except Exception as exc:
                last_error = exc
                raw_text = f"{type(exc).__name__}: {exc}"
                status = "error"
                if attempt >= args.max_retries:
                    break
                sleep_seconds = max(0.0, args.retry_backoff_sec) * attempt
                print(
                    f"failed to process {video_path} on attempt {attempt}/{args.max_retries}: {repr(exc)}; "
                    f"retrying in {sleep_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_seconds)

        if last_error is not None and status == "error":
            print(f"failed to process {video_path}: {repr(last_error)}", flush=True)

        parsed = {
            "path": path_value,
            "video_path": video_path,
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
    type_predictions_df.to_csv(type_predictions_path, index=False, lineterminator="\n")

    description = (
        f"Model: {args.model_name_or_path}\n"
        f"Jobs: {len(results)}\n"
        f"Elapsed: {total_elapsed:.2f}s\n"
        f"Video sampling override: disabled in this script\n"
        f"max_new_tokens: {args.max_new_tokens}\n"
        f"temperature: {args.temperature}\n"
        f"top_p: {args.top_p}\n"
        f"max_retries: {args.max_retries}\n"
        f"retry_backoff_sec: {args.retry_backoff_sec}\n"
        f"base_submission_csv: {resolved_base_submission_csv}\n"
        f"fallback_to_existing_type: {args.fallback_to_existing_type}\n"
        f"status_counts: {json.dumps(status_counts, ensure_ascii=False)}\n"
    )
    with open(description_path, "w", encoding="utf-8") as handle:
        handle.write(description)

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
