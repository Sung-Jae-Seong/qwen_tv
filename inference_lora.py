import argparse
import ast
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPO_ROOT / "result"
OUTPUT_ROOT = REPO_ROOT / "output"

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv(REPO_ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TEST_PATH_FILE = REPO_ROOT / "dataset" / "test_video_path.txt"
DEFAULT_TRAIN_PATH_FILE = REPO_ROOT / "dataset" / "train_video_path.txt"
DEFAULT_EXPERIMENT_NAME = "lora_inference"


def iter_candidate_paths(path_value: str, extra_bases: Optional[Iterable[Path]] = None):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        yield path
        return

    seen = set()
    bases = [Path.cwd(), REPO_ROOT]
    if extra_bases:
        bases.extend(Path(base) for base in extra_bases)

    for base in bases:
        candidate = (base / path).resolve()
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            yield candidate


def resolve_existing_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> Path:
    for candidate in iter_candidate_paths(path_value, extra_bases=extra_bases):
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in iter_candidate_paths(path_value, extra_bases=extra_bases))
    raise FileNotFoundError(f"Path not found: {path_value}. Searched: {searched}")


def maybe_resolve_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> str:
    for candidate in iter_candidate_paths(path_value, extra_bases=extra_bases):
        if candidate.exists():
            return str(candidate)
    return path_value


def resolve_lora_model_path(path_value: Optional[str]) -> Path:
    if path_value:
        candidate = resolve_existing_path(path_value)
        if candidate.is_dir() and candidate.name == "final_lora_model":
            return candidate

        nested_candidate = candidate / "final_lora_model"
        if nested_candidate.is_dir():
            return nested_candidate

        raise FileNotFoundError(
            "LoRA adapter directory not found. "
            f"Expected `{candidate}` or `{nested_candidate}`."
        )

    candidates = []
    direct_candidate = OUTPUT_ROOT / "final_lora_model"
    if direct_candidate.is_dir():
        candidates.append(direct_candidate)

    if OUTPUT_ROOT.exists():
        candidates.extend(
            path for path in OUTPUT_ROOT.glob("*/final_lora_model") if path.is_dir()
        )

    if not candidates:
        raise FileNotFoundError(
            "No LoRA adapter directory found under `output/`. "
            "Pass `--lora_model_path` explicitly."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_gpu_ids(requested_gpu_ids: Optional[list[int]]) -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device not found. `inference_lora.py` expects at least one visible NVIDIA GPU."
        )

    visible_gpu_count = torch.cuda.device_count()
    visible_gpu_ids = list(range(visible_gpu_count))

    if not requested_gpu_ids:
        return visible_gpu_ids

    deduped_gpu_ids = []
    for gpu_id in requested_gpu_ids:
        if gpu_id not in deduped_gpu_ids:
            deduped_gpu_ids.append(gpu_id)

    invalid_gpu_ids = [gpu_id for gpu_id in deduped_gpu_ids if gpu_id not in visible_gpu_ids]
    if invalid_gpu_ids:
        raise ValueError(
            f"Requested GPU IDs {invalid_gpu_ids} are not visible. "
            f"Visible GPU IDs: {visible_gpu_ids}"
        )

    return deduped_gpu_ids


def build_max_memory_map(gpu_ids: list[int], max_memory_per_gpu: Optional[str]) -> Optional[dict[int, str]]:
    if not max_memory_per_gpu:
        return None
    return {gpu_id: max_memory_per_gpu for gpu_id in gpu_ids}


def parse_in_json(llm_response, video_path):
    import re

    default = {
        "time": 0.0,
        "coordinate": [[0, 0], [0, 0]],
        "type": "head-on",
        "why": "invalid response format",
    }

    temp = None
    raw = llm_response

    cleaned = llm_response.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if code_block:
        cleaned = code_block.group(1).strip()

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            temp = json.loads(json_str)
        except json.JSONDecodeError:
            pass
        if temp is None:
            try:
                temp = ast.literal_eval(json_str)
            except Exception:
                pass

    if temp is None:
        try:
            temp = json.loads(llm_response)
        except Exception:
            pass
    if temp is None:
        try:
            temp = ast.literal_eval(llm_response)
        except Exception:
            pass

    if not isinstance(temp, dict):
        temp = default

    save_dir = RESULT_ROOT / "parsed_json"
    save_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    save_path = save_dir / f"{video_name}.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"failed to save parsed json for {video_path}: {e}", flush=True)

    raw_dir = RESULT_ROOT / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(raw_dir / f"{video_name}.txt", "w", encoding="utf-8") as f:
            f.write(raw)
    except Exception:
        pass

    return temp


def save_submission(submission, experiment_name, description_text="", submission_filename="submission.csv"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = RESULT_ROOT / f"{experiment_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    submission_path = save_dir / submission_filename
    description_path = save_dir / "description.txt"

    submission.to_csv(submission_path, index=False, lineterminator="\n")

    with open(description_path, "w", encoding="utf-8") as f:
        f.write(description_text)

    print(f"saved to {save_dir}")


def make_submission(results):
    rows = []

    for result in results:
        try:
            video_path = result.get("video_path", "")
            path = "videos/" + video_path.split("/videos/")[-1] if "/videos/" in video_path else video_path

            accident_time = result.get("time", 0.0)
            coordinate = result.get("coordinate", [[0, 0], [0, 0]])
            (x1, y1), (x2, y2) = coordinate

            center_x = round(((x1 + x2) / 2) / 1000, 3)
            center_y = round(((y1 + y2) / 2) / 1000, 3)
            accident_type = result.get("type", "unknown")

            rows.append(
                {
                    "path": path,
                    "accident_time": round(accident_time, 2),
                    "center_x": center_x,
                    "center_y": center_y,
                    "type": accident_type,
                }
            )
        except Exception as e:
            print(f"failed to make submission for result: {result}, error: {e}")
            continue

    submission = pd.DataFrame(
        rows,
        columns=["path", "accident_time", "center_x", "center_y", "type"],
    )
    return submission


class VideoInferenceVLM:
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


def load_video_paths(video_path=None, video_list=None, source="test"):
    if video_path:
        resolved_video_path = resolve_existing_path(video_path)
        return [str(resolved_video_path)], None

    if video_list:
        target_list = resolve_existing_path(video_list)
    else:
        target_list = DEFAULT_TRAIN_PATH_FILE if source == "train" else DEFAULT_TEST_PATH_FILE

    if not target_list.exists():
        raise FileNotFoundError(f"Video list file not found: {target_list}")

    video_paths = []
    with open(target_list, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            resolved = maybe_resolve_path(stripped, extra_bases=[target_list.parent])
            video_paths.append(resolved)

    return video_paths, str(target_list)


class Qwen3VLLoRAInference(VideoInferenceVLM):
    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        lora_model_path=None,
        gpu_ids=None,
        max_memory_per_gpu=None,
    ):
        self.gpu_ids = resolve_gpu_ids(gpu_ids)
        self.lora_model_path = resolve_lora_model_path(lora_model_path)
        self.max_memory = build_max_memory_map(self.gpu_ids, max_memory_per_gpu)

        print(f"Loading base model from: {model_id}")
        print(f"Using visible GPU IDs: {self.gpu_ids}")
        print(f"Using LoRA path: {self.lora_model_path}")

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.max_memory is not None:
            model_kwargs["max_memory"] = self.max_memory

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
            print("✓ Base model loaded successfully with flash_attention_2!")
        except Exception as e:
            print(f"Error loading model with flash_attention_2: {e}")
            print("Retrying without flash attention...")
            fallback_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=fallback_dtype,
                    **model_kwargs,
                )
                print(f"✓ Base model loaded successfully (dtype={fallback_dtype})!")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}") from e2

        self.device = next(self.model.parameters()).device
        print(f"Input device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self._apply_lora_manually()
        self.model.eval()

    def _apply_lora_manually(self):
        print(f"Applying LoRA weights manually from: {self.lora_model_path}")
        config_path = self.lora_model_path / "adapter_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"LoRA config not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            lora_config = json.load(f)

        r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("lora_alpha", 8)
        scaling = lora_alpha / r

        safetensors_path = self.lora_model_path / "adapter_model.safetensors"
        bin_path = self.lora_model_path / "adapter_model.bin"

        if safetensors_path.exists():
            lora_state_dict = load_file(safetensors_path)
        elif bin_path.exists():
            lora_state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("LoRA weights (.safetensors or .bin) not found.")

        lora_A = {}
        lora_B = {}
        for key, tensor in lora_state_dict.items():
            base_key = key.replace("base_model.model.", "")

            if "lora_A" in base_key:
                module_name = base_key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                lora_A[module_name] = tensor
            elif "lora_B" in base_key:
                module_name = base_key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                lora_B[module_name] = tensor

        model_state_dict = self.model.state_dict()
        updated_count = 0

        with torch.no_grad():
            for module_name in lora_A.keys():
                if module_name not in lora_B:
                    continue

                target_weight_key = f"{module_name}.weight"
                if target_weight_key not in model_state_dict:
                    print(f"Warning: Target layer '{target_weight_key}' not found in base model.")
                    continue

                W = model_state_dict[target_weight_key]
                A = lora_A[module_name].to(device=W.device, dtype=torch.float32)
                B = lora_B[module_name].to(device=W.device, dtype=torch.float32)

                delta_W = (B @ A) * scaling
                W.add_(delta_W.to(W.dtype))
                updated_count += 1

        print(f"✓ Successfully merged {updated_count} LoRA weight matrices into base model.")

    def video_inference(self, video_path, prompt, max_new_tokens=128):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL LoRA Inference")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory or its parent output directory. If omitted, auto-detects the newest output/*/final_lora_model.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["test", "train"],
        default="test",
        help="Dataset split to run when --video_list is not provided",
    )
    parser.add_argument("--video_path", type=str, default=None, help="Single video path")
    parser.add_argument("--video_list", type=str, default=None, help="Path to a text file containing video paths")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Output directory prefix")
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="Visible GPU IDs to use. Defaults to all visible GPUs. With CUDA_VISIBLE_DEVICES set, use local indices like 0 1.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Optional per-GPU memory cap such as 22GiB or 24000MiB.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate per video")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    instruction = (
        "return the time when the collision occurs, "
        "and return the collision bounding box with left-top and right-bottom coordinates. "
        "And return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, "
        "and t-bone collisions. "
        "Head-on is defined as a collision where the front ends of two vehicles hit each other. "
        "Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle. "
        "Sideswipe is defined as a slight collision where the sides of two vehicles hit each other. "
        "Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle. "
        "T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape."
    )

    return_format = """
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "time": exact time, the temporal location of the video where collision occured,
    "coordinate": left-top and right-bottom, the position of bounding box on the video frame that contains the collision,
    "type": choose and return one of the following [head-on, rear-end, sideswipe, single, t-bone],
    "why": explain why did you return that time, coordinate and type.
}
---
example:
{
    "time": "second.milisecond", # do not return time in hh:mm:ss format, for example, if the collision occurs at 1 second and 500 milliseconds, please return 1.5
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "one of the following [head-on, rear-end, sideswipe, single, t-bone]",
    "why": "..."
}
"""

    prompt = instruction + return_format
    video_paths, resolved_video_list_path = load_video_paths(
        video_path=args.video_path,
        video_list=args.video_list,
        source=args.source,
    )

    print(f"Found {len(video_paths)} videos")
    print(f"Source: {args.source}")
    if resolved_video_list_path:
        print(f"Video list: {resolved_video_list_path}")

    try:
        inference = Qwen3VLLoRAInference(
            model_id=args.model_id,
            lora_model_path=args.lora_model_path,
            gpu_ids=args.gpu_ids,
            max_memory_per_gpu=args.max_memory_per_gpu,
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback

        traceback.print_exc()
        return

    resolved_lora_path = str(inference.lora_model_path)
    results = []
    total_start_time = time.time()

    for i, video_path in enumerate(video_paths):
        try:
            print(f"\n{'=' * 60}")
            print(f"Processing {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
            print(f"{'=' * 60}")
            output = inference.video_inference(
                video_path,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            output_json = parse_in_json(output[0], video_path)
            output_json["video_path"] = video_path
            results.append(output_json)
            print("\n✓ Result:")
            print(json.dumps(output_json, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"\n✗ Error processing {video_path}: {e}")
            import traceback

            traceback.print_exc()

    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time

    print(f"\n{'=' * 60}")
    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds")
    print(f"Successfully processed {len(results)} videos")
    print(f"{'=' * 60}")

    if results:
        submission = make_submission(results)
        submission_filename = "predictions_train.csv" if args.source == "train" else "submission.csv"
        default_list_path = DEFAULT_TRAIN_PATH_FILE if args.source == "train" else DEFAULT_TEST_PATH_FILE
        description = (
            f"Source: {args.source}\n"
            f"Model: {args.model_id}\n"
            f"LoRA model: {resolved_lora_path}\n"
            f"GPU IDs: {inference.gpu_ids}\n"
            f"Max new tokens: {args.max_new_tokens}\n"
            f"Total videos: {len(results)}\n"
            f"Elapsed: {total_elapsed_seconds:.2f}s\n"
            f"Video list: {resolved_video_list_path or default_list_path}\n"
            f"Output file: {submission_filename}\n"
        )
        save_submission(
            submission,
            args.experiment_name,
            description,
            submission_filename=submission_filename,
        )


if __name__ == "__main__":
    main()
