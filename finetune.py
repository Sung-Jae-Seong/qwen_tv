"""
Fine-tune Qwen3-VL model with LoRA for accident detection SFT task.

This script uses QLoRA (4-bit quantization) to fine-tune Qwen3-VL-8B
on the CCTV accident detection dataset.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "dataset" / "sim_dataset"
DEFAULT_TRAIN_FILE = DATASET_ROOT / "train_sft.jsonl"
DEFAULT_VAL_FILE = DATASET_ROOT / "val_sft.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "qwen3_vl_lora"

load_dotenv(REPO_ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def str2bool(value):
    """Parse common string forms of booleans for argparse."""
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
    bases = [Path.cwd(), REPO_ROOT]
    if extra_bases:
        bases.extend(Path(base) for base in extra_bases)

    for base in bases:
        candidate = (base / path).resolve()
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            yield candidate


def find_existing_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> Optional[Path]:
    for candidate in iter_candidate_paths(path_value, extra_bases=extra_bases):
        if candidate.exists():
            return candidate
    return None


def resolve_existing_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> Path:
    resolved = find_existing_path(path_value, extra_bases=extra_bases)
    if resolved is not None:
        return resolved

    searched = ", ".join(str(path) for path in iter_candidate_paths(path_value, extra_bases=extra_bases))
    raise FileNotFoundError(f"Path not found: {path_value}. Searched: {searched}")


def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def maybe_resolve_path(path_value: str, extra_bases: Optional[Iterable[Path]] = None) -> str:
    resolved = find_existing_path(path_value, extra_bases=extra_bases)
    return str(resolved) if resolved is not None else path_value


def get_preferred_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    train_file: str = field(
        default=str(DEFAULT_TRAIN_FILE),
        metadata={"help": "Path to training JSONL file"},
    )
    val_file: str = field(
        default=str(DEFAULT_VAL_FILE),
        metadata={"help": "Path to validation JSONL file"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    num_frames: int = field(
        default=8,
        metadata={"help": "Number of frames to extract from video"},
    )


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={"help": "Model name or path"},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for fine-tuning"},
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Use QLoRA (4-bit quantization)"},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Target modules for LoRA"},
    )


class VideoSFTDataset(torch.utils.data.Dataset):
    """Dataset for video SFT training."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        num_frames: int = 8,
        max_seq_length: int = 2048,
    ):
        self.jsonl_path = resolve_existing_path(jsonl_path)
        self.processor = processor
        self.num_frames = num_frames
        self.max_seq_length = max_seq_length
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None
        self.base_dirs = [self.jsonl_path.parent]

        self.samples = []
        missing_videos = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                sample = json.loads(line)
                video_path = sample.get("video")
                resolved_video = maybe_resolve_path(video_path, extra_bases=self.base_dirs) if video_path else None
                sample["_resolved_video"] = resolved_video
                self.samples.append(sample)

                if resolved_video is None or not Path(resolved_video).exists():
                    missing_videos.append(video_path)

        print(f"Loaded {len(self.samples)} samples from {self.jsonl_path}")
        if missing_videos:
            print(
                f"Warning: {len(missing_videos)} samples reference missing video files. "
                "They will fall back to dummy frames."
            )
            preview = ", ".join(str(path) for path in missing_videos[:3])
            print(f"Missing video examples: {preview}")

    def extract_frames(self, video_path: str, num_frames: int = 8) -> Optional[List[np.ndarray]]:
        """Extract frames from video."""
        try:
            if not Path(video_path).exists():
                return None

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None

            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if len(frames) < num_frames:
                while len(frames) < num_frames:
                    frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))

            return frames[:num_frames]

        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = sample.get("_resolved_video") or sample["video"]
        messages = sample["messages"]

        frames = self.extract_frames(video_path, self.num_frames)
        if frames is None or len(frames) == 0:
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            inputs = self.processor(
                text=[text],
                images=None,
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )

            input_ids = (
                inputs["input_ids"].squeeze(0)
                if "input_ids" in inputs and inputs["input_ids"].numel() > 0
                else torch.tensor([2], dtype=torch.long)
            )
            attention_mask = inputs.get("attention_mask", torch.ones(len(input_ids))).squeeze(0).long()

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            prompt_messages = [message for message in messages if message["role"] != "assistant"]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=None,
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            prompt_ids = prompt_inputs["input_ids"].squeeze(0)
            prompt_length = len(prompt_ids)
            if prompt_length > 0:
                labels[:prompt_length] = -100

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
                if key in inputs and inputs[key] is not None:
                    value = inputs[key]
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        result[key] = value.squeeze(0) if value.dim() > 1 else value

            return result

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            if self.tokenizer:
                tokens = self.tokenizer("", return_tensors="pt")
                input_ids = tokens["input_ids"].squeeze(0)
                return {
                    "input_ids": input_ids,
                    "attention_mask": tokens["attention_mask"].squeeze(0).long(),
                    "labels": input_ids.clone(),
                }

            return {
                "input_ids": torch.tensor([2], dtype=torch.long),
                "attention_mask": torch.tensor([1], dtype=torch.long),
                "labels": torch.tensor([2], dtype=torch.long),
            }


def setup_model_and_processor(model_args: ModelArguments):
    """Setup model with LoRA and processor."""

    compute_dtype = get_preferred_compute_dtype()
    print(f"Model compute dtype: {compute_dtype}")

    if model_args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    return model, processor


class DataCollatorForVideoSFT:
    """Collate function for video+text SFT batches with proper padding."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_input_length = max(
            item["input_ids"].shape[0] if isinstance(item["input_ids"], torch.Tensor) else len(item["input_ids"])
            for item in batch
        )

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        video_tensors = {}

        for item in batch:
            input_ids = item["input_ids"] if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"])
            attention_mask = item.get("attention_mask", torch.ones_like(input_ids))
            labels = item.get("labels", input_ids.clone())

            if input_ids.shape[0] < max_input_length:
                pad_len = max_input_length - input_ids.shape[0]
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

            for key in ["pixel_values", "image_grid_thw", "video_grid_thw", "image_embeds", "video_embeds"]:
                if key in item and item[key] is not None:
                    video_tensors.setdefault(key, []).append(item[key])

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

        for key, values in video_tensors.items():
            if len(values) != len(batch):
                continue
            try:
                if key in ["pixel_values", "image_embeds", "video_embeds"]:
                    result[key] = torch.cat(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
                elif key in ["image_grid_thw", "video_grid_thw"]:
                    values_2d = [value.unsqueeze(0) if value.dim() == 1 else value for value in values]
                    result[key] = torch.cat(values_2d, dim=0)
                else:
                    result[key] = torch.stack(values)
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Could not batch {key}: {e}")

        return result


class SFTTrainer(Trainer):
    """Custom trainer for SFT with video input."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--val_file", type=str, default=str(DEFAULT_VAL_FILE))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    args = parser.parse_args()

    train_file = resolve_existing_path(args.train_file)
    val_file = resolve_existing_path(args.val_file)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_bf16 = args.bf16
    if effective_bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        print("Warning: bf16 was requested but is not supported in the current CUDA environment. Falling back to bf16=False.")
        effective_bf16 = False

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. QLoRA training may fail or be impractically slow on CPU.")

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=(
            args.lora_target_modules
            if args.lora_target_modules is not None
            else ModelArguments.__dataclass_fields__["lora_target_modules"].default_factory()
        ),
    )

    data_args = DataArguments(
        train_file=str(train_file),
        val_file=str(val_file),
        num_frames=args.num_frames,
        max_seq_length=args.max_seq_length,
    )

    print("=" * 50)
    print("Training configuration")
    print("=" * 50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Train JSONL: {train_file}")
    print(f"Val JSONL: {val_file}")
    print(f"Output dir: {output_dir}")
    print(f"Use QLoRA: {model_args.use_qlora}")
    print(f"Use LoRA: {model_args.use_lora}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}")
    print("=" * 50)

    print("Loading model and processor...")
    model, processor = setup_model_and_processor(model_args)

    print("\n" + "=" * 50)
    print("Loading datasets...")
    print("=" * 50)

    train_dataset = VideoSFTDataset(
        jsonl_path=data_args.train_file,
        processor=processor,
        num_frames=data_args.num_frames,
        max_seq_length=data_args.max_seq_length,
    )

    val_dataset = VideoSFTDataset(
        jsonl_path=data_args.val_file,
        processor=processor,
        num_frames=data_args.num_frames,
        max_seq_length=data_args.max_seq_length,
    )

    print("\n" + "=" * 50)
    print("Setting up training...")
    print("=" * 50)

    data_collator = DataCollatorForVideoSFT(
        pad_token_id=processor.tokenizer.pad_token_id or 0
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if args.max_steps < 0 else min(args.eval_steps, args.max_steps),
        save_steps=args.save_steps if args.max_steps < 0 else min(args.save_steps, args.max_steps),
        max_steps=args.max_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=effective_bf16,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    trainer.train()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    if model_args.use_lora:
        final_lora_dir = output_dir / "final_lora_model"
        model.save_pretrained(final_lora_dir)
        print(f"LoRA model saved to {final_lora_dir}")


if __name__ == "__main__":
    main()
