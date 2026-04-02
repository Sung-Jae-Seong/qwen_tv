#!/usr/bin/env python3
"""
Base Qwen/Qwen3.5-9B full-video two-stage classification inference.

Stage 1:
- single
- multi

Stage 2:
- head-on
- rear-end
- sideswipe
- t-bone

Stage 2 runs only for samples that Stage 1 classifies as "multi".
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
# Reduce allocator fragmentation during long two-stage GPU runs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference_qwen35_9b_base_video_classification import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL_ID,
    DEFAULT_RETRY_BACKOFF_SEC,
    DEFAULT_TEST_PATH_FILE,
    ensure_run_dir,
    extract_json_dict,
    generate_and_decode,
    load_jobs,
    maybe_expand_video_grid_for_generation,
    move_inputs_to_device,
    replace_type_in_submission,
    resolve_generation_device,
    sanitize_processor_kwargs,
    str2bool,
    summarize_status_counts,
)


DEFAULT_EXPERIMENT_NAME = "qwen35_9b_base_video_classification_two_stage"
STAGE1_ASSISTANT_JSON_PREFIX = '{"group": "'
STAGE2_ASSISTANT_JSON_PREFIX = '{"type": "'
FINAL_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone", "single"]

STAGE1_SYSTEM_PROMPT = """You are an AI vision expert for CCTV traffic accident analysis.

Your Stage 1 task is to classify the First Harmful Event in the video into exactly one of:

- multi
- single

Apply the crash classification standard based on ANSI D16.1 and FMCSA-2022.

[CRITICAL RULE: FOCUS ONLY ON THE FIRST HARMFUL EVENT]
Judge the event only at the exact moment of the actual crash impact.
Do not classify from precursors such as sudden braking, swerving, near-misses, loss of control, or post-impact secondary crashes.

[CORE CONCEPT: VEHICLE STATUS]
In-Transport:
A vehicle is in-transport if it is moving or stopped within the portion of the roadway ordinarily used for travel by similar vehicles.
This also includes a disabled vehicle abandoned in an active travel lane.

Not In-Transport:
A vehicle is not in-transport if it is not within the active traveled portion of the roadway.
This includes vehicles legally parked off the roadway, or vehicles completely stopped on the shoulder, median, or roadside.

[STAGE 1 LABEL MEANING]

- multi:
Use multi when the First Harmful Event IS a crash between two in-transport motor vehicles.
Representative multi examples include head-on, rear-end, sideswipe, t-bone, and any other first crash where one in-transport motor vehicle strikes another in-transport motor vehicle.
Output multi only when it is clearly visible that:
    - the second struck entity is another motor vehicle,
    - that motor vehicle is in-transport at the moment of impact,
    - and this contact is the First Harmful Event itself.
If those three points are clear, output multi even when the exact subtype (head-on, rear-end, sideswipe, or t-bone), struck surface, or impact geometry is brief, partially occluded, or uncertain.

- single:
Use single only when the First Harmful Event is NOT a crash between two in-transport motor vehicles.
This includes:
    - a one-vehicle crash,
    - a vehicle striking a fixed object,
    - a vehicle striking a pedestrian or non-motor-vehicle object,
    - a vehicle crashing into a not in-transport vehicle such as a legally parked vehicle or a vehicle completely stopped on the shoulder.

[DECISION PROCEDURE]

1. Identify the exact First Harmful Event.
2. Ask first: Did one motor vehicle crash with another motor vehicle at that moment?
3. If yes, decide whether both motor vehicles were in-transport at the First Harmful Event.
4. Output multi only if it is clear that the First Harmful Event is a crash between two in-transport motor vehicles.
5. Output single if it is unclear whether the second struck entity is a motor vehicle, unclear whether it is in-transport, or unclear whether the contact is the First Harmful Event rather than a later secondary crash.
6. Do not output single merely because the multi subtype is hard to determine.
7. If another motor vehicle in the active traveled roadway is clearly struck at the First Harmful Event, do not output single.

[RULES]

- This stage decides only whether the First Harmful Event is a two in-transport motor-vehicle crash.
- Subtype uncertainty is acceptable for multi, but vehicle-status uncertainty is not.
- Ignore later secondary crashes.
- Output exactly one label.
- Output without any additional information.

Output format:
{"group": "single | multi"}"""

STAGE1_USER_PROMPT = (
    "Analyze the full video. Output multi only when it is clear that the First Harmful Event is a crash between two "
    "in-transport motor vehicles. If the exact subtype is uncertain but the two in-transport motor-vehicle crash is clear, "
    "still output multi. If it is unclear whether the second struck entity is a motor vehicle, unclear whether it is "
    "in-transport, or unclear whether the contact is the First Harmful Event itself, output single. "
    "Return the required JSON only."
)

STAGE2_SYSTEM_PROMPT = """You are a multi-vehicle accident type classification expert.

Your only task is to classify the first primary crash into exactly one of:

- head-on
- rear-end
- sideswipe
- t-bone

Input assumption:
This clip is centered on the predicted accident location and shows the first primary crash or the frames immediately around it.
The accident has already been determined to involve a crash with another in-transport motor vehicle.
Therefore, do not predict single in this stage.

Reference definition for exclusion:

- single:
The First Harmful Event was not a crash with a motor vehicle in-transport.
Use this concept only to exclude cases that are not vehicle-to-vehicle crashes.
Do not output single in this stage.

Class definitions:

- t-bone:
A crash where the front of one motor vehicle impacts the side of another motor vehicle.
- head-on:
Front-to-front crash where the front end of one vehicle crashes into the front end of another vehicle while the two vehicles are traveling in opposite directions.
- rear-end:
Front-to-rear or rear-to-front crash where the initial contact point in the First Harmful Event is the front of one vehicle and the rear of the other vehicle.
- sideswipe:
Two vehicles traveling in the same or opposite direction make glancing contact.
The initial engagement does not significantly involve the front or rear surface areas.
Instead, the impact swipes along the vehicle surface roughly parallel to the direction of travel.

Decision rules:

- Focus only on the First Harmful Event.
- Classify from the initial contact geometry.
- Prefer struck surfaces and relative travel directions over pre-crash motion alone.
- Output head-on only when the initial contact is front-to-front and the vehicles are traveling in opposite directions.
- Output rear-end when the initial contact is front-to-rear or front-to-rear-corner, even if the vehicles later rotate or scrape along the side.
- Output sideswipe only when the initial contact is primarily side-to-side glancing or scraping contact.
- Do not output sideswipe if a front or rear surface is the main initial contact area.
- Do not output head-on for front-into-side impacts.
- Distinguish t-bone from sideswipe carefully:
t-bone is a more direct front-into-side crash,
while sideswipe is a glancing crash that continues along the side surface.

Rules:

- Choose exactly one label.
- Output without any additional information.

Output format:
{"type": "head-on | rear-end | sideswipe | t-bone"}"""

STAGE2_USER_PROMPT = (
    "Analyze the video and return the required JSON only. Focus on the First Harmful Event and classify from the "
    "initial contact geometry: front-to-front is head-on, front-to-rear is rear-end, side-to-side glancing contact "
    "is sideswipe, and direct front-into-side impact is t-bone."
)

STAGE1_DIRECT_MAP = {
    "single": "single",
    "singlevehicle": "single",
    "singlevehiclecrash": "single",
    "nonsingle": "multi",
    "notsingle": "multi",
    "nonsinglecrash": "multi",
    "multivehicle": "multi",
    "multivehiclecrash": "multi",
    "multi": "multi",
    "multiple": "multi",
}

STAGE1_PATTERNS = [
    ("multi", re.compile(r"\bnon[\s-]?single\b|\bnot[\s-]?single\b|\bmulti(?:-?vehicle)?\b|\bmultiple\b", re.IGNORECASE)),
    ("single", re.compile(r"\bsingle(?:[\s-]?vehicle(?:\s+crash)?)?\b", re.IGNORECASE)),
]

STAGE2_DIRECT_MAP = {
    "headon": "head-on",
    "rearend": "rear-end",
    "sideswipe": "sideswipe",
    "tbone": "t-bone",
}

STAGE2_PATTERNS = [
    ("head-on", re.compile(r"\bhead[\s-]?on\b", re.IGNORECASE)),
    ("rear-end", re.compile(r"\brear[\s-]?end\b", re.IGNORECASE)),
    ("sideswipe", re.compile(r"\bsideswipe\b", re.IGNORECASE)),
    ("t-bone", re.compile(r"\bt[\s-]?bone\b", re.IGNORECASE)),
]


def build_messages(
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    assistant_json_prefix: str,
    clip_start: Optional[float] = None,
    clip_end: Optional[float] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    video_item: Dict[str, Any] = {"type": "video", "video": video_path}
    if clip_start is not None:
        video_item["video_start"] = float(clip_start)
    if clip_end is not None:
        video_item["video_end"] = float(clip_end)
    if video_fps is not None:
        video_item["fps"] = float(video_fps)
    if video_max_frames is not None:
        video_item["max_frames"] = int(video_max_frames)

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                video_item,
                {"type": "text", "text": user_prompt},
            ],
        },
        {"role": "assistant", "content": assistant_json_prefix},
    ]


def normalize_with_patterns(value: Any, direct_map: Dict[str, str], patterns: List[Any]) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    direct_key = re.sub(r"[^a-z]+", "", text.lower())
    if direct_key in direct_map:
        return direct_map[direct_key]

    matches = []
    for label, pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((match.start(), label))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def maybe_restore_prefilled_json_for_stage(text: str, assistant_json_prefix: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return text
    return assistant_json_prefix + text


def extract_label_from_json(text: str, normalizer: Callable[[Any], Optional[str]]) -> Optional[str]:
    parsed = extract_json_dict(text)
    if not isinstance(parsed, dict):
        return normalizer(text)

    for key in ("group", "type", "label", "classification", "crash_type"):
        normalized = normalizer(parsed.get(key))
        if normalized is not None:
            return normalized

    answer = parsed.get("answer")
    if isinstance(answer, dict):
        for key in ("group", "type", "which", "label", "classification", "crash_type"):
            normalized = normalizer(answer.get(key))
            if normalized is not None:
                return normalized
    return None


def extract_stage_output(text: str, normalizer: Callable[[Any], Optional[str]]) -> Dict[str, Any]:
    parsed = extract_json_dict(text)

    return {
        "structured_output": parsed if isinstance(parsed, dict) else None,
        "label": extract_label_from_json(text, normalizer),
        "why": "",
    }


def normalize_stage1_label(value: Any) -> Optional[str]:
    return normalize_with_patterns(value, STAGE1_DIRECT_MAP, STAGE1_PATTERNS)


def normalize_stage2_label(value: Any) -> Optional[str]:
    return normalize_with_patterns(value, STAGE2_DIRECT_MAP, STAGE2_PATTERNS)


class Qwen35BaseVideoTwoStageRunner:
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

    def infer(
        self,
        *,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
    assistant_json_prefix: str,
    clip_start: Optional[float],
    clip_end: Optional[float],
    video_fps: Optional[float],
    video_max_frames: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
        messages = build_messages(
            video_path,
            system_prompt,
            user_prompt,
            assistant_json_prefix,
            clip_start=clip_start,
            clip_end=clip_end,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
        )
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


def run_stage_with_retries(
    *,
    runner: Qwen35BaseVideoTwoStageRunner,
    stage_name: str,
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    assistant_json_prefix: str,
    clip_start: Optional[float],
    clip_end: Optional[float],
    video_fps: Optional[float],
    video_max_frames: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    retry_backoff_sec: float,
    normalizer: Callable[[Any], Optional[str]],
) -> Dict[str, Any]:
    raw_text = ""
    last_error: Optional[Exception] = None
    structured_output = None
    label = ""
    why = ""
    status = "ok"

    for attempt in range(1, max_retries + 1):
        try:
            raw_text = runner.infer(
                video_path=video_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                assistant_json_prefix=assistant_json_prefix,
                clip_start=clip_start,
                clip_end=clip_end,
                video_fps=video_fps,
                video_max_frames=video_max_frames,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw_text = maybe_restore_prefilled_json_for_stage(raw_text, assistant_json_prefix)
            details = extract_stage_output(raw_text, normalizer)
            structured_output = details["structured_output"]
            label = details["label"] or ""
            why = details["why"]
            if not label:
                status = "parse_failed"
            break
        except Exception as exc:
            last_error = exc
            raw_text = f"{type(exc).__name__}: {exc}"
            status = "error"
            if attempt >= max_retries:
                break
            sleep_seconds = max(0.0, retry_backoff_sec) * attempt
            print(
                f"{stage_name} failed for {video_path} on attempt {attempt}/{max_retries}: {repr(exc)}; "
                f"retrying in {sleep_seconds:.1f}s",
                flush=True,
            )
            time.sleep(sleep_seconds)

    if last_error is not None and status == "error":
        print(f"{stage_name} failed for {video_path}: {repr(last_error)}", flush=True)

    return {
        "status": status,
        "raw_text": raw_text,
        "label": label,
        "why": why,
        "structured_output": structured_output,
    }


def save_two_stage_artifacts(
    run_dir: Path,
    path_value: str,
    stage1_raw_text: str,
    stage2_raw_text: str,
    parsed: Dict[str, Any],
):
    stem = Path(path_value).stem or "unknown"
    with open(run_dir / "raw_responses" / f"{stem}_stage1.txt", "w", encoding="utf-8") as handle:
        handle.write(stage1_raw_text)
    if stage2_raw_text:
        with open(run_dir / "raw_responses" / f"{stem}_stage2.txt", "w", encoding="utf-8") as handle:
            handle.write(stage2_raw_text)
    with open(run_dir / "parsed_predictions" / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)


def release_runner(runner: Optional["Qwen35BaseVideoTwoStageRunner"]) -> None:
    if runner is None:
        return

    try:
        del runner
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


def make_results_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        results,
        columns=[
            "path",
            "video_path",
            "clip_center_time",
            "clip_start",
            "clip_end",
            "predicted_type",
            "status",
            "stage1_label",
            "stage1_status",
            "stage1_why",
            "stage1_structured_output",
            "stage1_raw_output",
            "stage2_label",
            "stage2_status",
            "stage2_why",
            "stage2_structured_output",
            "stage2_raw_output",
        ],
    )


def make_stage1_only_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    stage1_rows = []
    for item in results:
        stage1_rows.append(
            {
                "path": item.get("path", ""),
                "video_path": item.get("video_path", ""),
                "clip_center_time": item.get("clip_center_time"),
                "clip_start": item.get("clip_start"),
                "clip_end": item.get("clip_end"),
                "stage1_group": item.get("stage1_label", ""),
                "stage1_status": item.get("stage1_status", ""),
                "stage1_raw_output": item.get("stage1_raw_output", ""),
                "final_status": item.get("status", ""),
                "predicted_type": item.get("predicted_type", ""),
            }
        )
    return pd.DataFrame(stage1_rows)


def attach_stage1_group_to_submission(base_submission_df: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    stage1_by_path = {
        str(item.get("path", "")).strip(): {
            "stage1_group": item.get("stage1_label", ""),
            "stage1_status": item.get("stage1_status", ""),
            "clip_center_time": item.get("clip_center_time"),
            "clip_start": item.get("clip_start"),
            "clip_end": item.get("clip_end"),
        }
        for item in results
    }

    output_df = base_submission_df.copy()
    output_df["stage1_group"] = [
        stage1_by_path.get(str(path).strip(), {}).get("stage1_group", "")
        for path in output_df["path"]
    ]
    output_df["stage1_status"] = [
        stage1_by_path.get(str(path).strip(), {}).get("stage1_status", "")
        for path in output_df["path"]
    ]
    output_df["clip_center_time"] = [
        stage1_by_path.get(str(path).strip(), {}).get("clip_center_time", "")
        for path in output_df["path"]
    ]
    output_df["clip_start"] = [
        stage1_by_path.get(str(path).strip(), {}).get("clip_start", "")
        for path in output_df["path"]
    ]
    output_df["clip_end"] = [
        stage1_by_path.get(str(path).strip(), {}).get("clip_end", "")
        for path in output_df["path"]
    ]
    return output_df


def map_stage1_label_to_submission_type(label: Any) -> str:
    text = str(label or "").strip().lower()
    if text == "single":
        return "single"
    if text == "multi":
        return "multi"
    return ""


def replace_type_in_submission_with_stage1_group(
    base_submission_df: pd.DataFrame,
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    stage1_type_by_path = {
        str(item.get("path", "")).strip(): map_stage1_label_to_submission_type(item.get("stage1_label", ""))
        for item in results
    }

    output_df = base_submission_df.copy()
    output_df["type"] = [
        stage1_type_by_path.get(str(path).strip(), "")
        for path in output_df["path"]
    ]
    return output_df


def apply_submission_time_clip_to_jobs(
    jobs: List[Dict[str, Any]],
    base_submission_df: Optional[pd.DataFrame],
    *,
    use_submission_time_clip: bool,
    clip_seconds_before: float,
    clip_seconds_after: float,
) -> int:
    if not use_submission_time_clip or base_submission_df is None:
        return 0
    if "accident_time" not in base_submission_df.columns:
        return 0

    accident_time_by_path: Dict[str, float] = {}
    for row in base_submission_df.to_dict("records"):
        path_value = str(row.get("path", "")).strip()
        if not path_value:
            continue
        raw_accident_time = row.get("accident_time")
        if pd.isna(raw_accident_time):
            continue
        try:
            accident_time = float(raw_accident_time)
        except (TypeError, ValueError):
            continue
        accident_time_by_path[path_value] = accident_time

    clipped = 0
    for job in jobs:
        path_value = str(job.get("path", "")).strip()
        accident_time = accident_time_by_path.get(path_value)
        if accident_time is None:
            continue
        clip_start = max(0.0, accident_time - max(0.0, clip_seconds_before))
        clip_end = accident_time + max(0.0, clip_seconds_after)
        if clip_end <= clip_start:
            clip_end = clip_start + 0.1
        job["clip_center_time"] = accident_time
        job["clip_start"] = clip_start
        job["clip_end"] = clip_end
        clipped += 1
    return clipped


def _clean_csv_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    return value


def _coerce_optional_float(value: Any) -> Optional[float]:
    value = _clean_csv_value(value)
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_existing_stage1_predictions(stage1_predictions_csv: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(stage1_predictions_csv)
    by_path: Dict[str, Dict[str, Any]] = {}

    for row in df.to_dict("records"):
        path_value = str(_clean_csv_value(row.get("path", ""))).strip()
        if not path_value:
            continue

        raw_text = str(_clean_csv_value(row.get("stage1_raw_output", "")))
        label_source = row.get("stage1_label", row.get("stage1_group", row.get("type", "")))
        label = normalize_stage1_label(_clean_csv_value(label_source)) or ""
        status = str(_clean_csv_value(row.get("stage1_status", ""))).strip()
        if not status:
            status = "ok" if label else "missing"

        structured_output = None
        structured_value = _clean_csv_value(row.get("stage1_structured_output", ""))
        if isinstance(structured_value, dict):
            structured_output = structured_value
        elif structured_value not in ("", None):
            structured_output = extract_json_dict(str(structured_value))
        if structured_output is None and label:
            structured_output = {"group": label}

        by_path[path_value] = {
            "label": label,
            "status": status,
            "why": str(_clean_csv_value(row.get("stage1_why", ""))),
            "structured_output": structured_output,
            "raw_text": raw_text,
            "previous_predicted_type": str(_clean_csv_value(row.get("predicted_type", ""))),
            "previous_final_status": str(_clean_csv_value(row.get("status", row.get("final_status", "")))),
            "clip_center_time": _coerce_optional_float(row.get("clip_center_time")),
            "clip_start": _coerce_optional_float(row.get("clip_start")),
            "clip_end": _coerce_optional_float(row.get("clip_end")),
        }
    return by_path


def parse_args():
    parser = argparse.ArgumentParser(description="Base Qwen3.5-9B full-video two-stage classification inference")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--stage1_model_name_or_path", type=str, default=None)
    parser.add_argument("--stage2_model_name_or_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_list", type=str, default=str(DEFAULT_TEST_PATH_FILE))
    parser.add_argument("--base_submission_csv", type=str, default=None)
    parser.add_argument("--stage1_predictions_csv", type=str, default=None)
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--stage1_max_new_tokens", type=int, default=192)
    parser.add_argument("--stage2_max_new_tokens", type=int, default=256)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_max_frames", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_backoff_sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--use_submission_time_clip", type=str2bool, default=True)
    parser.add_argument("--clip_seconds_before", type=float, default=3.0)
    parser.add_argument("--clip_seconds_after", type=float, default=3.0)
    parser.add_argument("--fallback_to_existing_type", type=str2bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    stage1_model_name_or_path = args.stage1_model_name_or_path or args.model_name_or_path
    stage2_model_name_or_path = args.stage2_model_name_or_path or args.model_name_or_path
    jobs, base_submission_df, resolved_base_submission_csv = load_jobs(
        video_path=args.video_path,
        video_list=args.video_list,
        base_submission_csv=args.base_submission_csv,
        limit=args.limit,
    )
    clipped_job_count = apply_submission_time_clip_to_jobs(
        jobs,
        base_submission_df,
        use_submission_time_clip=args.use_submission_time_clip,
        clip_seconds_before=args.clip_seconds_before,
        clip_seconds_after=args.clip_seconds_after,
    )
    if not jobs:
        raise ValueError("No jobs found. Check --video_path, --video_list, or --base_submission_csv.")

    run_dir = ensure_run_dir(args.experiment_name)
    type_predictions_path = run_dir / "type_predictions.csv"
    stage1_predictions_path = run_dir / "stage1_group_predictions.csv"
    description_path = run_dir / "description.txt"

    print("=" * 80, flush=True)
    print("Qwen3.5-9B base full-video two-stage classification inference", flush=True)
    print("=" * 80, flush=True)
    print(f"Stage 1 model: {stage1_model_name_or_path}", flush=True)
    print(f"Stage 2 model: {stage2_model_name_or_path}", flush=True)
    print("Stage 1 labels: single | multi", flush=True)
    print("Stage 2 labels: head-on | rear-end | sideswipe | t-bone", flush=True)
    print("Stage 1 input: full video", flush=True)
    print("Stage 2 input: clipped video around base submission accident_time when available", flush=True)
    print(f"Video fps override: {args.video_fps}", flush=True)
    print(f"Video max_frames override: {args.video_max_frames}", flush=True)
    print(f"Stage 2 submission-time clip enabled: {args.use_submission_time_clip}", flush=True)
    print(f"Stage 2 clip window: -{args.clip_seconds_before:.2f}s / +{args.clip_seconds_after:.2f}s", flush=True)
    print(f"Jobs with stage2 submission-time clip: {clipped_job_count}", flush=True)
    print(f"Jobs: {len(jobs)}", flush=True)
    if resolved_base_submission_csv is not None:
        print(f"Base submission CSV: {resolved_base_submission_csv}", flush=True)
    if args.stage1_predictions_csv:
        print(f"Stage 1 predictions CSV: {args.stage1_predictions_csv}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)

    total_start_time = time.time()
    results: List[Dict[str, Any]] = []
    type_by_path: Dict[str, str] = {}
    multi_indices: List[int] = []
    if args.stage1_predictions_csv:
        print("Stage 1 mode: skipped. Reusing existing stage1 predictions.", flush=True)
        stage1_by_path = load_existing_stage1_predictions(args.stage1_predictions_csv)
        print(f"Loaded stage1 predictions for {len(stage1_by_path)} paths.", flush=True)

        for index, job in enumerate(jobs, start=1):
            path_value = job["path"]
            video_path = job["video_path"]
            clip_center_time = job.get("clip_center_time")
            clip_start = job.get("clip_start")
            clip_end = job.get("clip_end")
            prior_stage1 = stage1_by_path.get(path_value, {})
            stage1 = {
                "label": prior_stage1.get("label", ""),
                "status": prior_stage1.get("status", "missing"),
                "why": prior_stage1.get("why", ""),
                "structured_output": prior_stage1.get("structured_output"),
                "raw_text": prior_stage1.get("raw_text", ""),
            }
            stage2 = {
                "status": "not_run",
                "raw_text": "",
                "label": "",
                "why": "",
                "structured_output": None,
            }
            predicted_type = ""
            final_status = stage1["status"]

            print(f"\n{'=' * 60}", flush=True)
            print(f"Stage 1 (reused) {index}/{len(jobs)}: {Path(video_path).name}", flush=True)
            print(f"{'=' * 60}", flush=True)

            if stage1["status"] == "ok":
                if stage1["label"] == "single":
                    predicted_type = "single"
                    final_status = "ok"
                elif stage1["label"] == "multi":
                    final_status = "pending_stage2"
                else:
                    final_status = "stage1_parse_failed"
            elif stage1["status"] in {"parse_failed", "stage1_parse_failed"}:
                final_status = "stage1_parse_failed"
            elif stage1["status"] == "missing":
                final_status = "stage1_missing"
            else:
                final_status = "stage1_error"

            parsed = {
                "path": path_value,
                "video_path": video_path,
                "clip_center_time": clip_center_time,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "predicted_type": predicted_type,
                "status": final_status,
                "stage1_label": stage1["label"],
                "stage1_status": stage1["status"],
                "stage1_why": stage1["why"],
                "stage1_structured_output": stage1["structured_output"],
                "stage1_raw_output": stage1["raw_text"],
                "stage2_label": stage2["label"],
                "stage2_status": stage2["status"],
                "stage2_why": stage2["why"],
                "stage2_structured_output": stage2["structured_output"],
                "stage2_raw_output": stage2["raw_text"],
            }
            results.append(parsed)
            save_two_stage_artifacts(
                run_dir=run_dir,
                path_value=path_value,
                stage1_raw_text=stage1["raw_text"],
                stage2_raw_text=stage2["raw_text"],
                parsed=parsed,
            )
            if final_status == "pending_stage2":
                multi_indices.append(len(results) - 1)
            else:
                type_by_path[path_value] = predicted_type
    else:
        print("Sequential model loading: stage1 first, then stage2 for multi samples only", flush=True)
        print("Loading stage1 runner...", flush=True)
        stage1_runner = Qwen35BaseVideoTwoStageRunner(
            model_name_or_path=stage1_model_name_or_path,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
        )

        for index, job in enumerate(jobs, start=1):
            path_value = job["path"]
            video_path = job["video_path"]
            clip_center_time = job.get("clip_center_time")
            clip_start = job.get("clip_start")
            clip_end = job.get("clip_end")
            print(f"\n{'=' * 60}", flush=True)
            print(f"Stage 1 {index}/{len(jobs)}: {Path(video_path).name}", flush=True)
            print(f"{'=' * 60}", flush=True)
            print("Using full video for stage1.", flush=True)

            stage1 = run_stage_with_retries(
                runner=stage1_runner,
                stage_name="stage1",
                video_path=video_path,
                system_prompt=STAGE1_SYSTEM_PROMPT,
                user_prompt=STAGE1_USER_PROMPT,
                assistant_json_prefix=STAGE1_ASSISTANT_JSON_PREFIX,
                clip_start=None,
                clip_end=None,
                video_fps=args.video_fps,
                video_max_frames=args.video_max_frames,
                max_new_tokens=args.stage1_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
                normalizer=normalize_stage1_label,
            )

            stage2 = {
                "status": "not_run",
                "raw_text": "",
                "label": "",
                "why": "",
                "structured_output": None,
            }
            predicted_type = ""
            final_status = stage1["status"]

            if stage1["status"] == "ok":
                if stage1["label"] == "single":
                    predicted_type = "single"
                    final_status = "ok"
                elif stage1["label"] == "multi":
                    final_status = "pending_stage2"
                else:
                    final_status = "stage1_parse_failed"
            elif stage1["status"] == "parse_failed":
                final_status = "stage1_parse_failed"
            else:
                final_status = "stage1_error"

            parsed = {
                "path": path_value,
                "video_path": video_path,
                "clip_center_time": clip_center_time,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "predicted_type": predicted_type,
                "status": final_status,
                "stage1_label": stage1["label"],
                "stage1_status": stage1["status"],
                "stage1_why": stage1["why"],
                "stage1_structured_output": stage1["structured_output"],
                "stage1_raw_output": stage1["raw_text"],
                "stage2_label": stage2["label"],
                "stage2_status": stage2["status"],
                "stage2_why": stage2["why"],
                "stage2_structured_output": stage2["structured_output"],
                "stage2_raw_output": stage2["raw_text"],
            }
            results.append(parsed)
            save_two_stage_artifacts(
                run_dir=run_dir,
                path_value=path_value,
                stage1_raw_text=stage1["raw_text"],
                stage2_raw_text=stage2["raw_text"],
                parsed=parsed,
            )
            if final_status == "pending_stage2":
                multi_indices.append(len(results) - 1)
            else:
                type_by_path[path_value] = predicted_type

            print("stage1 raw:", flush=True)
            print(stage1["raw_text"], flush=True)
            print(
                json.dumps(
                    {
                        "path": path_value,
                        "predicted_type": predicted_type,
                        "status": final_status,
                        "stage1_label": stage1["label"],
                        "stage1_status": stage1["status"],
                        "stage2_label": stage2["label"],
                        "stage2_status": stage2["status"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

        print("\nReleasing stage1 runner and clearing CUDA cache...", flush=True)
        release_runner(stage1_runner)

    if multi_indices:
        print(f"Loading stage2 runner for {len(multi_indices)} multi samples...", flush=True)
        stage2_runner = Qwen35BaseVideoTwoStageRunner(
            model_name_or_path=stage2_model_name_or_path,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
        )

        for multi_order, result_index in enumerate(multi_indices, start=1):
            record = results[result_index]
            video_path = record["video_path"]
            path_value = record["path"]
            clip_start = record.get("clip_start")
            clip_end = record.get("clip_end")
            clip_center_time = record.get("clip_center_time")
            print(f"\n{'=' * 60}", flush=True)
            print(f"Stage 2 {multi_order}/{len(multi_indices)}: {Path(video_path).name}", flush=True)
            print(f"{'=' * 60}", flush=True)
            if clip_start is not None and clip_end is not None:
                print(
                    f"Using clipped window from base submission time: center={clip_center_time:.3f}s, "
                    f"start={clip_start:.3f}s, end={clip_end:.3f}s",
                    flush=True,
                )
            else:
                print("No base submission accident_time clip available; using full video for stage2.", flush=True)

            stage2 = run_stage_with_retries(
                runner=stage2_runner,
                stage_name="stage2",
                video_path=video_path,
                system_prompt=STAGE2_SYSTEM_PROMPT,
                user_prompt=STAGE2_USER_PROMPT,
                assistant_json_prefix=STAGE2_ASSISTANT_JSON_PREFIX,
                clip_start=clip_start,
                clip_end=clip_end,
                video_fps=args.video_fps,
                video_max_frames=args.video_max_frames,
                max_new_tokens=args.stage2_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
                normalizer=normalize_stage2_label,
            )

            predicted_type = ""
            final_status = "stage2_error"
            if stage2["status"] == "ok" and stage2["label"] in {"head-on", "rear-end", "sideswipe", "t-bone"}:
                predicted_type = stage2["label"]
                final_status = "ok"
            elif stage2["status"] == "parse_failed":
                final_status = "stage2_parse_failed"

            record["predicted_type"] = predicted_type
            record["status"] = final_status
            record["stage2_label"] = stage2["label"]
            record["stage2_status"] = stage2["status"]
            record["stage2_why"] = stage2["why"]
            record["stage2_structured_output"] = stage2["structured_output"]
            record["stage2_raw_output"] = stage2["raw_text"]

            if predicted_type:
                type_by_path[path_value] = predicted_type
            elif path_value not in type_by_path:
                type_by_path[path_value] = ""

            save_two_stage_artifacts(
                run_dir=run_dir,
                path_value=path_value,
                stage1_raw_text=record["stage1_raw_output"],
                stage2_raw_text=record["stage2_raw_output"],
                parsed=record,
            )

            print("stage2 raw:", flush=True)
            print(stage2["raw_text"], flush=True)
            print(
                json.dumps(
                    {
                        "path": path_value,
                        "predicted_type": predicted_type,
                        "status": final_status,
                        "stage1_label": record["stage1_label"],
                        "stage1_status": record["stage1_status"],
                        "stage2_label": record["stage2_label"],
                        "stage2_status": record["stage2_status"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

        print("\nReleasing stage2 runner and clearing CUDA cache...", flush=True)
        release_runner(stage2_runner)
    else:
        print("No samples were classified as multi in stage1; stage2 was skipped.", flush=True)

    total_elapsed = time.time() - total_start_time
    status_counts = summarize_status_counts(results)

    print(f"\nTotal elapsed time: {total_elapsed:.2f} seconds", flush=True)
    print(f"Completed {len(results)} jobs with statuses: {json.dumps(status_counts, ensure_ascii=False)}", flush=True)

    results_df = make_results_df(results)
    results_df.to_csv(type_predictions_path, index=False, lineterminator="\n")
    stage1_only_df = make_stage1_only_df(results)
    stage1_only_df.to_csv(stage1_predictions_path, index=False, lineterminator="\n")

    description = (
        f"Stage 1 model: {stage1_model_name_or_path}\n"
        f"Stage 2 model: {stage2_model_name_or_path}\n"
        f"Jobs: {len(results)}\n"
        f"Elapsed: {total_elapsed:.2f}s\n"
        f"Pipeline: two-stage sequential loading (stage1=single|multi, stage2=head-on|rear-end|sideswipe|t-bone)\n"
        f"stage1_input: full video\n"
        f"stage2_input: clipped around base submission accident_time when available\n"
        f"video_fps: {args.video_fps}\n"
        f"video_max_frames: {args.video_max_frames}\n"
        f"use_submission_time_clip: {args.use_submission_time_clip}\n"
        f"clip_seconds_before: {args.clip_seconds_before}\n"
        f"clip_seconds_after: {args.clip_seconds_after}\n"
        f"clipped_job_count: {clipped_job_count}\n"
        f"stage1_max_new_tokens: {args.stage1_max_new_tokens}\n"
        f"stage2_max_new_tokens: {args.stage2_max_new_tokens}\n"
        f"temperature: {args.temperature}\n"
        f"top_p: {args.top_p}\n"
        f"max_retries: {args.max_retries}\n"
        f"retry_backoff_sec: {args.retry_backoff_sec}\n"
        f"base_submission_csv: {resolved_base_submission_csv}\n"
        f"stage1_predictions_csv: {args.stage1_predictions_csv}\n"
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
        stage1_submission_df = replace_type_in_submission_with_stage1_group(base_submission_df, results)
        stage1_submission_out_path = run_dir / "submission_replace_only_type_stage1.csv"
        stage1_submission_df.to_csv(stage1_submission_out_path, index=False, lineterminator="\n")
        print(f"Saved replaced submission to: {submission_out_path}", flush=True)
        print(f"Saved stage1-only replaced submission to: {stage1_submission_out_path}", flush=True)

    print(f"Saved type predictions to: {type_predictions_path}", flush=True)
    print(f"Saved stage1-only predictions to: {stage1_predictions_path}", flush=True)
    print(f"Saved run artifacts to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
