"""
Fine-tune Qwen3-VL model with LoRA for accident detection SFT task.

This script uses QLoRA (4-bit quantization) to fine-tune Qwen3-VL-8B
on the CCTV accident detection dataset.
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_file: str = field(
        default="dataset/sim_dataset/train_sft.jsonl",
        metadata={"help": "Path to training JSONL file"}
    )
    val_file: str = field(
        default="dataset/sim_dataset/val_sft.jsonl",
        metadata={"help": "Path to validation JSONL file"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    num_frames: int = field(
        default=8,
        metadata={"help": "Number of frames to extract from video"}
    )


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-8b-Instruct",
        metadata={"help": "Model name or path"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for fine-tuning"}
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Use QLoRA (4-bit quantization)"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
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
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.num_frames = num_frames
        self.max_seq_length = max_seq_length
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else None
        
        # Load JSONL
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> Optional[List[np.ndarray]]:
        """Extract frames from video."""
        try:
            if not os.path.exists(video_path):
                return None
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            # Select frame indices uniformly
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
                # Pad with last frame
                while len(frames) < num_frames:
                    frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
            
            return frames[:num_frames]
        
        except Exception:
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        video_path = sample['video']
        messages = sample['messages']
        
        # Extract frames
        frames = self.extract_frames(video_path, self.num_frames)
        
        # Fallback to dummy frames if video can't be loaded
        if frames is None or len(frames) == 0:
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.num_frames)]
        
        # Process with Qwen processor
        try:
            # Create a string containing the chat format prompt without formatting prompt
            # The apply_chat_template will convert the message array into proper formatting
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Create input combining text and video
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            
            # Prepare output - use input_ids as labels (standard SFT approach)
            input_ids = inputs['input_ids'].squeeze(0) if 'input_ids' in inputs and inputs['input_ids'].numel() > 0 else torch.tensor([2], dtype=torch.long)
            attention_mask = inputs.get('attention_mask', torch.ones(len(input_ids))).squeeze(0).long()
            
            # Labels: -100 for padding, actual token ids elsewhere
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Mask user prompt in labels (so model only learns to predict assistant's response)
            prompt_messages = [m for m in messages if m['role'] != 'assistant']
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=None,
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            # Find the size of the prompt without padding
            prompt_ids = prompt_inputs['input_ids'].squeeze(0)
            prompt_length = len(prompt_ids)
            if prompt_length > 0:
                # Mask the prompt so we only train on assistant output
                labels[:prompt_length] = -100
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
            # Add pixel/video values if present
            for key in ['pixel_values', 'image_grid_thw', 'video_grid_thw']:
                if key in inputs and inputs[key] is not None:
                    val = inputs[key]
                    if isinstance(val, torch.Tensor) and val.numel() > 0:
                        result[key] = val.squeeze(0) if val.dim() > 1 else val
            
            return result
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return minimal valid batch
            if self.tokenizer:
                tokens = self.tokenizer("", return_tensors="pt")
                input_ids = tokens['input_ids'].squeeze(0)
                return {
                    'input_ids': input_ids,
                    'attention_mask': tokens['attention_mask'].squeeze(0).long(),
                    'labels': input_ids.clone(),
                }
            else:
                return {
                    'input_ids': torch.tensor([2], dtype=torch.long),
                    'attention_mask': torch.tensor([1], dtype=torch.long),
                    'labels': torch.tensor([2], dtype=torch.long),
                }


def setup_model_and_processor(model_args: ModelArguments):
    """Setup model with LoRA and processor."""
    
    if model_args.use_qlora:
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load model with QLoRA quantization
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load model with bfloat16 for standard LoRA
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Setup LoRA
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
    
    # Load processor
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
        # Process each item
        batch_inputs = {}
        
        # Get max length for padding
        max_input_length = max(
            item['input_ids'].shape[0] if isinstance(item['input_ids'], torch.Tensor) else len(item['input_ids'])
            for item in batch
        )
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # Separate video/image tensors
        video_tensors = {}
        
        for item in batch:
            input_ids = item['input_ids'] if isinstance(item['input_ids'], torch.Tensor) else torch.tensor(item['input_ids'])
            attention_mask = item.get('attention_mask', torch.ones_like(input_ids))
            labels = item.get('labels', input_ids.clone())
            
            # Pad sequences
            if input_ids.shape[0] < max_input_length:
                pad_len = max_input_length - input_ids.shape[0]
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            
            # Collect video/image tensors
            for key in ['pixel_values', 'image_grid_thw', 'video_grid_thw', 'image_embeds', 'video_embeds']:
                if key in item and item[key] is not None:
                    if key not in video_tensors:
                        video_tensors[key] = []
                    video_tensors[key].append(item[key])
        
        result = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
        }
        
        # Add video tensors with padding if necessary
        for key, values in video_tensors.items():
            if len(values) == len(batch):
                try:
                    if key in ['pixel_values', 'image_embeds', 'video_embeds']:
                        # These might have different shapes, need to pad or concatenate properly
                        result[key] = torch.cat(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
                    elif key in ['image_grid_thw', 'video_grid_thw']:
                        # Ensure 2D shape (num_items, 3) for THW
                        # If a single item is (3,), it should become (1, 3) before concat/stack
                        values_2d = [v.unsqueeze(0) if v.dim() == 1 else v for v in values]
                        result[key] = torch.cat(values_2d, dim=0)
                    else:
                        result[key] = torch.stack(values)
                except (RuntimeError, ValueError) as e:
                    print(f"Warning: Could not batch {key}: {e}")
                    pass
        
        return result


class SFTTrainer(Trainer):
    """Custom trainer for SFT with video input."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8b-Instruct")
    parser.add_argument("--train_file", type=str, default="dataset/sim_dataset/train_sft.jsonl")
    parser.add_argument("--val_file", type=str, default="dataset/sim_dataset/val_sft.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output/qwen3_vl_lora")
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
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--use_qlora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup model arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Setup data arguments
    data_args = DataArguments(
        train_file=args.train_file,
        val_file=args.val_file,
        num_frames=args.num_frames,
        max_seq_length=args.max_seq_length,
    )
    
    print("=" * 50)
    print("Loading model and processor...")
    print("=" * 50)
    
    model, processor = setup_model_and_processor(model_args)
    
    print("\n" + "=" * 50)
    print("Loading datasets...")
    print("=" * 50)
    
    # Load datasets
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
    
    # Data collator
    data_collator = DataCollatorForVideoSFT(
        pad_token_id=processor.tokenizer.pad_token_id or 0
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        bf16=args.bf16,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    # Create trainer
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
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    # Save model
    if model_args.use_lora:
        model.save_pretrained(os.path.join(args.output_dir, "final_lora_model"))
        print(f"LoRA model saved to {os.path.join(args.output_dir, 'final_lora_model')}")


if __name__ == "__main__":
    main()
